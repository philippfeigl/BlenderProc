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
    def __init__(self, file_paths, images, poses=None, scenes_path=None):
        self.file_paths=file_paths
        self.images=images
        self.poses=poses
        self.scenes_path=scenes_path
        
    def save_data(self, idx=None, idxs=None):
        if idx is not None:
            color_bgr = self.images[idx].copy()
            color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
            cv2.imwrite(self.file_paths[idx], color_bgr)
        elif idxs is not None:
            for curr_idx in idxs:
                color_bgr = self.images[curr_idx-1].copy()
                color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]
                cv2.imwrite(self.file_paths[curr_idx-1], color_bgr)
            np.save(osp.join(self.scenes_path, 'poses_groundtruth_time.npy'), self.poses)
            del self.file_paths
            del self.images
            del self.poses

def get_geometry_name():
    return f'h_{h_cone}_theta_{theta_cone}_rmax_{rmax}'

def get_dataset_paths(from_harddive):
    if from_harddive:
        media_path = '/media/philipp'
        harddrive_path = [os.path.join(media_path, entry) for entry in os.listdir(media_path)][0]
        servoing_path = os.path.join(harddrive_path, 'visual_servoing')
        local_servoing_path = os.path.join(os.getcwd(), 'src/visual_servoing')
        groundtruth_path = os.path.join(local_servoing_path, 'groundtruth')
        scenes_path = os.path.join(servoing_path, 'scenes')
        geometry_name = get_geometry_name()
        geometry_path = os.path.join(scenes_path, geometry_name)
        blender_name = 'dataset_blender'
        parameter_scenes_paths = [os.path.join(geometry_path, entry) for entry in os.listdir(geometry_path)]
    return parameter_scenes_paths, geometry_name, blender_name

def load_camera_info():
    path = get_camera_groundtruth().replace('groundtruth', 'camera_data')
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

def get_splitted_pose_lines(file_path):
    splitted_lines = None
    with open(file_path) as file:
        lines = file.read()
        splitted_lines = lines.split('\n')
    return splitted_lines

def get_target_pose(suffix='target'):
    cam_gt_path = get_camera_groundtruth()
    geometry_name = get_geometry_name()
    poses_path = osp.join(cam_gt_path, geometry_name)
    target_file = osp.join(poses_path, 'poses_' + suffix + '.txt')
    target_splitted = get_splitted_pose_lines(target_file)
    return target_splitted

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

def remove_cam_keyframes(frame:int):
    action = cam_obj.animation_data.action
    if action is None:
        return
    for fc in action.fcurves:
        print(f"{frame=}")
        cam_obj.keyframe_delete(data_path=fc.data_path, frame=frame)

remove_keyframes_vec = np.vectorize(remove_cam_keyframes)

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
parser.add_argument('cc_textures_path', default="resources/cctextures", help="Path to downloaded cc textures")
parser.add_argument('output_dir', help="Path to where the final files will be saved ")
parser.add_argument('--num_scenes', type=int, default=2, help="How many scenes with 25 images each to generate")
parser.add_argument('--ycbv_only', type=bool, default=True, help="Choose if only use ycbv models")
parser.add_argument('--fx', type=float, default=600, help='Focal length in x direction')
parser.add_argument('--fy', type=float, default=600, help='Focal length in y direction')
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
# bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# Intrinsics from realsense plugin for gazebo
# K = np.array([[462.1379699707031, 0.0, 320.0],
#               [0.0, 462.1379699707031, 240.0],
#               [0.0, 0.0, 1.0]])
K = np.array([[args.fx, 0, 320],[0, args.fy, 240],[0, 0, 1]])
bproc.camera.set_intrinsics_from_K_matrix(K, 640, 480)

all_parameter_scenes_paths, geometry_name, blender_name = get_dataset_paths(from_harddive=True)
# print(f"{bproc.camera.get_intrinsics_as_K_matrix()=}")

target_poses = get_target_pose('target')
target_poses_reversed = get_target_pose('target_reversed')

def add_cam_pose_by_line(line, cam_poses):
    current_pose = np.fromstring(line, dtype=np.float32, sep=',')[1:]
    location = current_pose[:3]# - np.array([table_center_x, table_center_y, table_center_z])
    rotation_matrix = R.from_quat(current_pose[3:]).as_matrix()
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    cam2world_opcv = transform_pose(cam2world_matrix)
    cam2world_blender = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world_opcv.copy(), ["X", "-Y", "-Z"])
    return cam2world_opcv.flatten(), cam2world_blender

sp_pt = 20
traj_forwards = 0
traj_backwards = 0
for i in range(args.num_scenes):
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

    for parameter_scenes_paths in all_parameter_scenes_paths:
        random_values = np.random.choice(np.arange(1, sp_pt+1), size=10, replace=False)
        random_values = np.hstack([random_values, np.random.choice(np.arange(sp_pt+1, 2*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(2*sp_pt+1, 3*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(3*sp_pt+1, 4*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(4*sp_pt+1, 5*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(5*sp_pt+1, 6*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(6*sp_pt+1, 7*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(7*sp_pt+1, 8*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(8*sp_pt+1, 9*sp_pt+1), size=10, replace=False)])
        random_values = np.hstack([random_values, np.random.choice(np.arange(9*sp_pt+1, 10*sp_pt+1), size=10, replace=False)])
        for num_traj in list(random_values):
            # sample CC Texture and assign to room planes
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)
            cam_poses = 0
            stored_rgb_files = []
            poses_array = np.zeros((0, 16))
            scene_path = os.path.join(parameter_scenes_paths, f'scene_{num_traj}')
            file_path = os.path.join(scene_path, 'poses_groundtruth_time.txt')
            if '_reversed' in parameter_scenes_paths:
                t_poses = target_poses_reversed
                traj_backwards += 1
                scene_string = f'scene_{traj_backwards}'
            else:
                t_poses = target_poses
                traj_forwards += 1
                scene_string = f'scene_{traj_forwards}'
            blender_scenes_path = scene_path.replace(geometry_name, blender_name).replace(f'scene_{num_traj}', scene_string)
            chunk_path = osp.join(blender_scenes_path, 'rgb')
            if not osp.exists(chunk_path):
                os.makedirs(chunk_path)
            # new_scenes_path = osp.join(osp.join(os.getcwd(), args.output_dir), f'scene_{k}')
            stored_rgb_files.append(osp.join(blender_scenes_path, 'target.png'))
            # 0th Etry is target
            cam2world_opcv, c2w_blender = add_cam_pose_by_line(t_poses[num_traj], cam_poses)
            bproc.camera.add_camera_pose(c2w_blender, frame=cam_poses)
            cam_poses += 1
            poses_array = np.vstack([poses_array, cam2world_opcv])
            if not osp.exists(blender_scenes_path):
                os.makedirs(blender_scenes_path)
            with open(file_path) as file:
                lines = file.read()
                splitted_lines = lines.split('\n')
                for j, line in enumerate(splitted_lines, 1):
                    if line.strip():
                        cam2world_opcv, c2w_blender = add_cam_pose_by_line(line, cam_poses)
                        bproc.camera.add_camera_pose(c2w_blender, frame=cam_poses)
                        cam_poses += 1
                        poses_array = np.vstack([poses_array, cam2world_opcv])
                        stored_rgb_files.append(os.path.join(chunk_path, f'{j:05d}.png'))
                # render the whole pipeline
                
                data = bproc.renderer.render()
                data_storer = DataStorer(stored_rgb_files, data['colors'], poses_array, blender_scenes_path)
                idxs = np.arange(1, j+1, 1)
                data_storer.save_data(idxs=idxs)
    
    for obj in (sampled_bop):      
        obj.disable_rigidbody()
        obj.hide(True)