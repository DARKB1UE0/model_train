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
# 💡 强烈建议：第一次先跑 10 张测试！确认标签和图片完美后，再改成 10000
NUM_IMAGES = 10 

# ⚠️ 请确保将 "your_username" 替换为您实际的 Linux 用户名（如果就是 shark 则无需修改）
OUTPUT_DIR = "/home/shark/model_train/dataset"
BG_IMAGES_DIR = "/home/shark/model_train/backgrounds" 
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
# 🌟 强制设置渲染分辨率为 640x640 (完美适配 YOLO 默认输入，防止被压缩变形)
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.resolution_percentage = 100

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
    light_main.location = (random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(2, 8))
    if light_main.data.type in ['POINT', 'SPOT']:
        light_main.data.energy = random.uniform(500, 2000)
    elif light_main.data.type == 'SUN':
         light_main.data.energy = random.uniform(1, 10)

    for obj in bpy.data.objects:
        if obj.name.startswith("Aux_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
            
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
        
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    composite = tree.nodes.new('CompositorNodeComposite')
    alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
    scale_node = tree.nodes.new('CompositorNodeScale')
    image_node = tree.nodes.new('CompositorNodeImage')
    
    scale_node.space = 'RENDER_SIZE'
    
    bg_path = random.choice(bg_images)
    img = bpy.data.images.load(bg_path)
    image_node.image = img
    
    tree.links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    tree.links.new(scale_node.outputs['Image'], alpha_over.inputs[1])
    tree.links.new(render_layers.outputs['Image'], alpha_over.inputs[2])
    tree.links.new(alpha_over.outputs['Image'], composite.inputs['Image'])

def randomize_camera_pose(camera, target_obj):
    """相机球面漫游采样，并强制对准矿石"""
    radius = random.uniform(0.5, 2.5)          # 距离 0.5m ~ 2.5m
    azimuth = random.uniform(0, 2 * math.pi)   # 方位角 0 ~ 360度
    elevation = random.uniform(0.1, math.pi/2) # 高度角，避免钻入地下

    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.cos(elevation) * math.sin(azimuth)
    z = radius * math.sin(elevation)
    camera.location = (x, y, z)

    constraint_name = "Auto_Track_To"
    track_constraint = camera.constraints.get(constraint_name)
    if not track_constraint:
        track_constraint = camera.constraints.new(type='TRACK_TO')
        track_constraint.name = constraint_name
        track_constraint.target = target_obj
        track_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_constraint.up_axis = 'UP_Y'

def randomize_ore_pose(ore_model, camera):
    """随机化矿石的 3D 旋转姿态，并随机偏移相机的 2D 画面中心"""
    # 1. 让矿石在 3D 空间中全向翻滚
    ore_model.rotation_euler = (
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi),
        random.uniform(0, 2 * math.pi)
    )
    
    # 2. 随机偏移相机镜头的 2D 中心，打破“目标永远居中”的魔咒
    # 偏移量限制在 -0.3 到 0.3 之间，防止矿石完全跑出画面外
    camera.data.shift_x = random.uniform(-0.3, 0.3)
    camera.data.shift_y = random.uniform(-0.3, 0.3)

# ==========================================
# 4. 标注生成逻辑 (Annotation)
# ==========================================
def get_yolo_annotation(scene, camera, empties):
    """计算 2D 投影并生成 YOLO-Pose 格式的字符串"""
    keypoints_2d = []
    x_coords = []
    y_coords = []
    
    for empty in empties:
        coords_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, empty.matrix_world.translation)
        
        # 翻转 Y 轴匹配 YOLO 坐标系
        x = coords_2d.x
        y = 1.0 - coords_2d.y
        
        keypoints_2d.append((x, y))
        x_coords.append(x)
        y_coords.append(y)
        
    # 计算 Bounding Box 并限制在 0~1 范围内
    x_min = max(0.0, min(x_coords))
    x_max = min(1.0, max(x_coords))
    y_min = max(0.0, min(y_coords))
    y_max = min(1.0, max(y_coords))
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
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
        # 1. 执行所有域随机化 (材质、光照、背景、相机位姿、矿石姿态)
        randomize_material(ore_model)
        randomize_lighting()
        setup_background_image()
        randomize_camera_pose(camera, ore_model)
        randomize_ore_pose(ore_model, camera)
        
        # 强制更新视图层，确保所有物理变换和约束生效
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
