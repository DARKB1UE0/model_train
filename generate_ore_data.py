import bpy
import bpy_extras
import math
import random
import os
import glob
from mathutils import Vector, Euler

# Configuration
NUM_IMAGES = 10000
OUTPUT_DIR = "/home/shark/model_train/dataset"
BG_IMAGES_DIR = "/home/shark/model_train/backgrounds" # User should place background images here
CLASS_ID = 0

# Ensure output directories exist
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

# Get objects
scene = bpy.context.scene
camera = bpy.data.objects.get("Camera")
ore_model = bpy.data.objects.get("Ore_Model")
light_main = bpy.data.objects.get("Light_Main")

# Keypoint empties
empties = [bpy.data.objects.get(f"Empty_P{i}") for i in range(8)]

# Basic checks
if not camera or not ore_model or not light_main or any(e is None for e in empties):
    print("Error: Missing required objects in the scene.")
    # In a real script, we might exit here.

# Camera setup (Pinhole, no distortion)
camera.data.type = 'PERSP'
camera.data.lens = 35  # 35mm focal length

def get_random_material_props():
    roughness = random.uniform(0.1, 0.9)
    metallic = random.uniform(0.0, 1.0)
    return roughness, metallic

def randomize_material(obj):
    if not obj.data.materials:
        # Create a basic material if none exists
        mat = bpy.data.materials.new(name="Ore_Material")
        mat.use_nodes = True
        obj.data.materials.append(mat)
    
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    
    if bsdf:
        roughness, metallic = get_random_material_props()
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic

def randomize_lighting():
    # Randomize main light
    light_main.location = (
        random.uniform(-5, 5),
        random.uniform(-5, 5),
        random.uniform(2, 8)
    )
    if light_main.data.type == 'POINT' or light_main.data.type == 'SPOT':
        light_main.data.energy = random.uniform(500, 2000)
    elif light_main.data.type == 'SUN':
         light_main.data.energy = random.uniform(1, 10)

    # Add 1-3 random auxiliary lights
    # First, clean up old auxiliary lights
    for obj in bpy.data.objects:
        if obj.name.startswith("Aux_Light"):
            bpy.data.objects.remove(obj, do_unlink=True)
            
    num_aux_lights = random.randint(1, 3)
    for i in range(num_aux_lights):
        light_data = bpy.data.lights.new(name=f"Aux_Light_Data_{i}", type='POINT')
        light_data.energy = random.uniform(100, 1000)
        light_data.color = (random.random(), random.random(), random.random())
        
        light_obj = bpy.data.objects.new(name=f"Aux_Light_{i}", object_data=light_data)
        scene.collection.objects.link(light_obj)
        
        light_obj.location = (
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(1, 5)
        )

def setup_background_image():
    # Ensure compositor is used
    scene.use_nodes = True
    tree = scene.node_tree
    
    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
        
    # Create necessary nodes
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    composite = tree.nodes.new('CompositorNodeComposite')
    alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
    scale_node = tree.nodes.new('CompositorNodeScale')
    scale_node.space = 'RENDER_SIZE'
    
    # Load random background image
    bg_images = glob.glob(os.path.join(BG_IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(BG_IMAGES_DIR, "*.png"))
    
    if bg_images:
        bg_path = random.choice(bg_images)
        img = bpy.data.images.load(bg_path)
        img_node = tree.nodes.new('CompositorNodeImage')
        img_node.image = img
        
        # Link nodes
        tree.links.new(img_node.outputs[0], scale_node.inputs[0])
        tree.links.new(scale_node.outputs[0], alpha_over.inputs[1])
        tree.links.new(render_layers.outputs[0], alpha_over.inputs[2])
        tree.links.new(alpha_over.outputs[0], composite.inputs[0])
        
        # Transparent film for background rendering
        scene.render.film_transparent = True
    else:
        # Fallback if no images found
        tree.links.new(render_layers.outputs[0], composite.inputs[0])
        scene.render.film_transparent = False

def randomize_camera_pose():
    # Spherical sampling
    r = random.uniform(3.0, 8.0) # Radius
    theta = random.uniform(0, math.pi / 2.5) # Polar angle (0 is zenith, don't go below horizon)
    phi = random.uniform(0, 2 * math.pi) # Azimuthal angle

    # Convert to Cartesian
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    
    camera.location = (x, y, z)
    
    # Track to origin (Ore_Model)
    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

def get_2d_keypoints():
    pts_2d = []
    for empty in empties:
        # Get 3D world coordinate
        coord_3d = empty.matrix_world.translation
        # Project to 2D normalized camera view
        coord_2d = bpy_extras.object_utils.world_to_camera_view(scene, camera, coord_3d)
        
        # Blender origin is bottom-left, YOLO origin is top-left
        pt_x = coord_2d.x
        pt_y = 1.0 - coord_2d.y
        pts_2d.append((pt_x, pt_y))
    return pts_2d

def generate_yolo_label(pts_2d, label_path):
    # Calculate Bounding Box
    x_coords = [p[0] for p in pts_2d]
    y_coords = [p[1] for p in pts_2d]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Center and Dimensions
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    
    # Clamp values to [0, 1]
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    # Format string
    label_str = f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
    
    for pt in pts_2d:
        label_str += f" {pt[0]:.6f} {pt[1]:.6f} 2"
        
    with open(label_path, 'w') as f:
        f.write(label_str)

# Main Loop
for i in range(NUM_IMAGES):
    print(f"Generating image {i}/{NUM_IMAGES}")
    
    # 1. Randomize Material
    randomize_material(ore_model)
    
    # 2. Randomize Lighting
    randomize_lighting()
    
    # 3. Setup Background
    setup_background_image()
    
    # 4. Randomize Camera
    randomize_camera_pose()
    
    # Update scene for projections
    bpy.context.view_layer.update()
    
    # 5. Render
    image_filename = f"ore_{i:05d}.jpg"
    image_path = os.path.join(OUTPUT_DIR, "images", image_filename)
    scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)
    
    # 6. Generate Labels
    pts_2d = get_2d_keypoints()
    label_filename = f"ore_{i:05d}.txt"
    label_path = os.path.join(OUTPUT_DIR, "labels", label_filename)
    generate_yolo_label(pts_2d, label_path)

print("Data generation complete!")
