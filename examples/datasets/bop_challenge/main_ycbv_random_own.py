import blenderproc as bproc
import argparse
import os
import os.path as osp
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import pickle
import cv2
import bpy

h_cone = 2
theta_cone = 45
rmax = 1
table_center_x = 0.21 # 0.25
table_center_y = -0.122 # -0.108 # -0.12
table_center_z = 0.76

class DataStorer():
    def __init__(self, file_paths, images, poses=None):
        self.file_paths=file_paths
        self.images=images
        self.poses=poses
        
    def save_data(self):
        try:
            for idx in range(len(self.file_paths)):
                print(f"{self.file_paths[idx]=}")
                cv2.imwrite(self.file_paths[idx], self.images[idx])
        except:
            print(f"AAAAAAAAAAAAHHHHHHH")

def get_dataset_paths(from_harddive):
    if from_harddive:
        media_path = '/media/philipp'
        harddrive_path = [os.path.join(media_path, entry) for entry in os.listdir(media_path)][0]
        servoing_path = os.path.join(harddrive_path, 'visual_servoing')
        local_servoing_path = os.path.join(os.getcwd(), 'src/visual_servoing')
        groundtruth_path = os.path.join(local_servoing_path, 'groundtruth')
        scenes_path = os.path.join(servoing_path, 'scenes')
        geometry_path = os.path.join(scenes_path, f'h_{h_cone}_theta_{theta_cone}_rmax_{rmax}')
        blender_scene_path = os.path.join(scenes_path, 'dataset_blender')
        parameter_scenes_paths = [os.path.join(geometry_path, entry) for entry in os.listdir(geometry_path)]
    return parameter_scenes_paths, blender_scene_path

def load_camera_info():
    path = get_camera_groundtruth().replace('groundtruth', 'camera_data')
    print(f"{path=}")
    # Read camera_info
    camera_info = None
    with open(os.path.join(path, 'camera_info'), 'rb') as file:
        # dump information to that file
        camera_info = pickle.load(file)
        # print(c_to_w)
    return camera_info

def get_camera_groundtruth():
    # Paths
    ws_path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(ws_path, '../../../../'))
    vs_path = os.path.join(parent_path, 'visual_servoing')
    groundtruth_path = os.path.join(vs_path, "groundtruth")
    return groundtruth_path

def get_base_to_optical(splitted_lines):
    bo = splitted_lines.split(',')
    base_to_opt = [np.array([bo[0], bo[1], bo[2]], dtype=np.float32),
                    np.array([bo[6], bo[3], bo[4], bo[5]], dtype=np.float32)]
    return base_to_opt

def get_transform_matrix():
    groundtruth_path = get_camera_groundtruth()
    groundtruth_file = 'groundtruth.txt'
    file_path = os.path.join(groundtruth_path, groundtruth_file)
    with open(file_path) as file:
        lines = file.read()
    base_to_opt = get_base_to_optical(lines)
    rot = R.from_quat(base_to_opt[1]).as_matrix()
    M_origin_to_camera = np.concatenate([rot, np.array([base_to_opt[0]]).T], axis=1)
    M_origin_to_camera = np.concatenate([M_origin_to_camera, np.array([[0, 0, 0, 1]])], axis=0)
    return M_origin_to_camera

def transform_pose(pose):
    transformation_matrix_base_to_camera = get_transform_matrix()
    return np.dot(pose, transformation_matrix_base_to_camera)

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2, help="How many scenes with 25 images each to generate")
parser.add_argument('--ycbv_only', type=bool, default=True, help="Choose if only use ycbv models")
args = parser.parse_args()

bproc.init()

# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'), mm2m = True)

# load distractor bop objects
if not args.ycbv_only:
    tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tless'), model_type = 'cad', mm2m = True)
    hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'hb'), mm2m = True)
    tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, 'tyol'), mm2m = True)
    bop_objects = (target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs)
else:
    bop_objects = (target_bop_objs)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, 'ycbv'))

# set shading and hide objects
for obj in bop_objects:
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
tx, ty, tz = table_center_x, table_center_y, table_center_z
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 0.01], location=[tx, ty, tz-0.01]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[tx, ty-2, tz+2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[tx, ty+2, tz+2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[tx+2, ty, tz+2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[tx-2, ty, tz+2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[tx, ty, tz+10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([tx-0.3, ty-0.3, tz+0.1], [tx-0.2, ty-0.2, tz+0.3])
    max = np.random.uniform([tx+0.2, ty+0.2, tz+0.4], [tx+0.3, ty+0.3, tz+0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

all_parameter_scenes_paths, blender_paths = get_dataset_paths(from_harddive=True)


for i in range(args.num_scenes):

    if i==1:
        K = np.array([[462.1379699707031, 0.0, 320.0],
                      [0.0, 462.1379699707031, 240.0],
                      [0.0, 0.0, 1.0]])
        bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)

    # Sample bop objects for a scene
    num_ycbv_objs = np.random.randint(low=1, high=21, size=1, dtype=int)[0]
    print(f"{num_ycbv_objs=}")
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=num_ycbv_objs, replace=False))
    sampled_bop = sampled_target_bop_objs
    if not args.ycbv_only:
        sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=2, replace=False))
        sampled_distractor_bop_objs += list(np.random.choice(hb_dist_bop_objs, size=2, replace=False))
        sampled_distractor_bop_objs += list(np.random.choice(tyol_dist_bop_objs, size=2, replace=False))
        sampled_bop += sampled_distractor_bop_objs
        sampled_bop_objs = (sampled_bop)
    else:
        sampled_bop_objs = (sampled_target_bop_objs)
    # Randomize materials and set physics
    for obj in sampled_bop_objs:        
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [tx, ty, tz], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_bop,
                              sample_pose_func = sample_pose_func, 
                              max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_bop)

    cam_poses = 0
    k = 0
    for parameter_scenes_paths in all_parameter_scenes_paths:
        num_trajectories = 2
        for num_traj in range(1, num_trajectories+1):
            if num_traj % 2 == 0:
                scene_path = os.path.join(parameter_scenes_paths, f'scene_{num_traj}')
                file_path = os.path.join(scene_path, 'poses_groundtruth_time.txt')
                with open(file_path) as file:
                    lines = file.read()
                    splitted_lines = lines.split('\n')
                    stored_rgb_files = []
                    for j, line in enumerate(splitted_lines, 1):
                        if j==5:
                            break
                        if line.strip():
                            current_pose = np.fromstring(line, dtype=np.float32, sep=',')[1:]
                            location = current_pose[:3]# - np.array([table_center_x, table_center_y, table_center_z])
                            rotation_matrix = R.from_quat(current_pose[3:]).as_matrix()
                            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
                            cam2world_matrix = transform_pose(cam2world_matrix)
                            cam2world_matrix = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world_matrix, ["X", "-Y", "-Z"])
                            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                            cam_poses += 1
                            if j==1:
                                chunk_path = osp.join(osp.join(os.getcwd(), args.output_dir), f'rgb_{k}')
                                if not os.path.exists(chunk_path):
                                    os.makedirs(chunk_path)
                            stored_rgb_files.append(os.path.join(chunk_path, f'{j:05d}.png'))
                    # render the whole pipeline
                    idxs = np.arange(1, j, 1)
                    data = bproc.renderer.render()
                    data_storer = DataStorer(stored_rgb_files, data['colors'])
                    data_storer.save_data()
                    
                    
        k += 1     
    
    for obj in (sampled_bop):      
        obj.disable_rigidbody()
        obj.hide(True)
    
    # render the whole pipeline
    # data = bproc.renderer.render()
    # bproc.image.save_images(os.path.join(args.output_dir), data['images']['RGB'])
    """
    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'ycbv',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10)
    
    for obj in (sampled_bop):      
        obj.disable_rigidbody()
        obj.hide(True)
    """