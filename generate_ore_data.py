import bpy
import bpy_extras
import math
import random
import os
import glob
from mathutils import Vector, Euler

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
NUM_IMAGES = 10000
OUTPUT_DIR = "/home/shark/model_train/dataset"
BG_IMAGES_DIR = "/home/shark/model_train/backgrounds" # 请确保此文件夹内有真实的背景图片
CLASS_ID = 0

# 确保输出目录存在
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# 获取场景和对象引用
scene = bpy.context.scene
camera = bpy.data.objects.get("Camera")
ore_model = bpy.data.objects.get("Ore_Model")
light_main = bpy.data.objects.get("Light_Main")

# 获取 8 个关键点 (Empty)
empties = [bpy.data.objects.get(f"Empty_P{i}") for i in range(8)]

# 基础检查
if not camera or not ore_model or not light_main or any(e is None for e in empties):
    raise ValueError("错误：场景中缺失必要的对象 (Camera, Ore_Model, Light_Main 或 Empty_P0~P7)。请检查您的 .blend 文件。")

# ==========================================
# 2. 初始化设置 (Initialization)
# ==========================================
# 相机设置为完美的针孔相机 (无畸变，用于训练 YOLO)
camera.data.type = 'PERSP'
camera.data.lens = 35  # 35mm 焦距

# 开启透明背景，以便合成器能将矿石叠加在背景图片上
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'JPEG'

# 获取所有背景图片路径
bg_images = glob.glob(os.path.join(BG_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(BG_IMAGES_DIR, "*.png"))
if not bg_images:
    print("警告：未找到背景图片，将使用透明/纯色背景渲染。")

# ==========================================
# 3. 域随机化功能组 (Domain Randomization)
# ==========================================
def randomize_material(obj):
    """随机化矿石材质的粗糙度和金属度"""
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Ore_Material")
        mat.use_nodes = True
        obj.data.materials.append(mat)
    
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    if bsdf:
        bsdf.inputs['Roughness'].default_value = random.uniform(0.1, 0.9)
        bsdf.inputs['Metallic'].default_value = random.uniform(0.0, 1.0)

def randomize_lighting():
    """随机化主光源和辅助光源"""
    # 随机化主光源位置和强度
    light_main.location = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(2, 8))
    if light_main.data.type in ['POINT', 'SPOT']:
        light_main.data.energy = random.uniform(500, 2000)
    elif light_main.data.type == 'SUN':
         light_main.data.energy = random.uniform(1, 10)

    # 清理旧的辅助光源
    for obj in bpy.data.objects:
        if obj.name.startswith("Aux_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
            
    # 随机生成 1-3 个辅助光源
    for i in range(random.randint(1, 3)):
        light_data = bpy.data.lights.new(name=f"Aux_Light_Data_{i}", type='POINT')
        light_data.energy = random.uniform(100, 1000)
        light_data.color = (random.random(), random.random(), random.random())
        
        light_obj = bpy.data.objects.new(name=f"Aux_Light_{i}", object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_obj.location = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(1, 5))

def setup_background_image():
    """在合成器中随机加载一张真实图片作为背景"""
    if not bg_images: return

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
        
    # 创建节点
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    composite = tree.nodes.new('CompositorNodeComposite')
    alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
    scale_node = tree.nodes.new('CompositorNodeScale')
    image_node = tree.nodes.new('CompositorNodeImage')
    
    # 设置缩放节点以适应渲染尺寸
    scale_node.space = 'RENDER_SIZE'
    
    # 随机加载图片
    bg_path = random.choice(bg_images)
    img = bpy.data.images.load(bg_path)
    image_node.image = img
    
    # 连接节点：Image -> Scale -> Alpha Over (背景层)；Render -> Alpha Over (前景层)
    tree.links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
    tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
    tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])

def randomize_camera_pose(camera, target_obj):
    """相机球面漫游采样，并强制对准矿石"""
    radius = random.uniform(0.5, 2.5)          # 距离
    azimuth = random.uniform(0, 2 * math.pi)   # 方位角 (360度)
    elevation = random.uniform(0.1, math.pi/2) # 高度角 (避免钻入地下)

    # 球面转笛卡尔坐标
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)
    camera.location = (x, y, z)

    # 强制对准约束
    constraint_name = "Auto_Track_To"
    track_constraint = camera.constraints.get(constraint_name)
    if not track_constraint:
        track_constraint = camera.constraints.new(type='TRACK_TO')
        track_constraint.name = constraint_name
        track_constraint.target = target_obj
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.up_axis = 'UP_Y'

# ==========================================
# 4. 标注生成逻辑 (Annotation)
# ==========================================
def get_yolo_annotation(scene, camera, empties):
    """计算 2D 投影并生成 YOLO-Pose 格式的字符串"""
    keypoints_2d = []
    x_coords = []
    y_coords = []
    
    for empty in empties:
        # 获取 3D 世界坐标到 2D 相机视图的归一化坐标
        coords_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, empty.matrix_world.translation)
        
        # Blender 原点在左下角，YOLO 原点在左上角，必须翻转 Y 轴！
        x = coords_2d.x
        y = 1.0 - coords_2d.y
        
        keypoints_2d.append((x, y))
        x_coords.append(x)
        y_coords.append(y)
        
    # 计算 Bounding Box (并将其限制在 0~1 范围内，防止越界报错)
    x_min = max(0.0, min(x_coords))
    x_max = min(1.0, max(x_coords))
    y_min = max(0.0, min(y_coords))
    y_max = min(1.0, max(y_coords))
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 组装 YOLO 字符串: class x_c y_c w h p0_x p0_y 2 p1_x p1_y 2 ...
    # 尾部的 '2' 代表关键点可见且已标注
    anno_str = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    for kp in keypoints_2d:
        anno_str += f" {kp[0]:.6f} {kp[1]:.6f} 2"
        
    return anno_str

# ==========================================
# 5. 主渲染循环 (Main Loop)
# ==========================================
if __name__ == "__main__":
    print(f"🚀 开始生成 {NUM_IMAGES} 张合成数据...")
    
    for i in range(NUM_IMAGES):
        # 1. 执行所有域随机化
        randomize_material(ore_model)
        randomize_lighting()
        setup_background_image()
        randomize_camera_pose(camera, ore_model)
        
        # 强制更新视图层，确保相机约束和坐标计算正确
        bpy.context.view_layer.update()
        
        # 2. 设置文件路径并渲染图像
        image_filename = f"ore_{i:05d}.jpg"
        scene.render.filepath = os.path.join(OUTPUT_DIR, "images", image_filename)
        bpy.ops.render.render(write_still=True)
        
        # 3. 计算并保存 YOLO 标签
        yolo_str = get_yolo_annotation(scene, camera, empties)
        label_filepath = os.path.join(OUTPUT_DIR, "labels", f"ore_{i:05d}.txt")
        
        with open(label_filepath, "w") as f:
            f.write(yolo_str + "\n")
            
        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"✅ 已完成: {i + 1} / {NUM_IMAGES}")

    print("🎉 数据集生成完毕！请前往 dataset 目录查看。")

