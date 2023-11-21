import viser
import viser.transforms as tf
import time
import numpy as np
import trimesh as tr
import tyro
from ur5py.ur5 import UR5Robot
from robot_lerf.ur5_motion_planning import UR5MotionPlanning
import tqdm

from robot_lerf.graspnet_baseline.load_ns_model import NerfstudioWrapper, RealsenseCamera
from robot_lerf.graspnet_baseline.graspnet_module import GraspNetModule
from autolab_core import RigidTransform
import open3d as o3d
import matplotlib
from robot_lerf.graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector
from robot_lerf.capture_utils import _generate_hemi,HOME_POS

UR5_HOME_JOINT=[-3.2835257689105433, -1.7120874563800257, 1.6925301551818848, -1.7154505888568323, -1.6483410040484827, 1.5793789625167847]

def get_bbox_from_grasp(gg):
    center = gg.translation
    R = gg.rotation_matrix
    H= np.eye(4)
    H[:3,:3] = R
    H[:3,3] = center
    extent=np.array((gg.depth,gg.width,gg.height))
    box = o3d.geometry.OrientedBoundingBox(center,H[:3,:3],extent)
    return box

def main(
    config_path: str,  # Nerfstudio model config path, of format outputs/.../config.yml
    graspnet_ckpt: str = 'robot_lerf/graspnet_baseline/logs/log_kn/checkpoint.tar',  # GraspNet checkpoint path
    ):
    # robot = UR5Robot()
    # robot.set_tcp(RigidTransform(rotation=np.eye(3),translation=np.array([0,0,0.16])))
    # time.sleep(.5)
    # robot.move_joint(HOME_POS,vel=.2)
    # motion_planner = UR5MotionPlanning()
    graspnet_large = GraspNetModule()
    graspnet_large.init_net(graspnet_ckpt, cylinder_radius=0.04)

    ns_wrapper = NerfstudioWrapper(config_path)
    world_pointcloud,global_pointcloud,table_center = ns_wrapper.create_pointcloud()

    server = viser.ViserServer()

    server.add_point_cloud(
        name=f"world_pointcloud",
        points=np.asarray(world_pointcloud.vertices),
        colors=np.asarray(world_pointcloud.visual.vertex_colors[:, :3]),
        point_size=0.002,
    )

    gen_grasp_button = server.add_gui_button(
        name=f"Generate GRASPS",
    )
    gen_grasp_text = server.add_gui_text(
        name=f"LERF query",
        initial_value="",
    )
    hemi_radius = server.add_gui_number(
        name=f"hemi_radius",
        initial_value=2,
    )
    hemi_theta_N = server.add_gui_number(
        name=f"hemi_theta_N",
        initial_value=15,
    )
    hemi_phi_N = server.add_gui_number(
        name=f"hemi_phi_N",
        initial_value=1,
    )
    hemi_th_range = server.add_gui_number(
        name=f"hemi_th_bounds",
        initial_value=90,
    )
    hemi_phi_down = server.add_gui_number(
        name=f"hemi_phi_down",
        initial_value=0,
    )
    hemi_phi_up = server.add_gui_number(
        name=f"hemi_phi_up",
        initial_value=0,
    )
    mfcdetector = ModelFreeCollisionDetector(global_pointcloud.vertices, voxel_size=0.005)
    def local_collision_detection(gg):
        collision_mask = mfcdetector.detect(gg, collision_thresh=0.000)
        gg = gg[~collision_mask]
        return gg

    grasp_list = []
    @gen_grasp_button.on_click
    def _(_):
        nonlocal grasp_list,world_pointcloud,global_pointcloud
        gen_grasp_button.disabled = True
        gen_grasp_text.disabled = True
        lerf_word = gen_grasp_text.value.split(";")
        ns_wrapper.pipeline.image_encoder.set_positives(lerf_word)

        for grasp in grasp_list:
            grasp.remove()#GuiHandle

        # Get the LERF activation pointcloud for the given query
        # TODO convert this to a dino floodfill-style thing

        # lerf_relevancy, lerf_xyz = [], []
        # for ind,cam_pose in enumerate(tqdm.tqdm(_generate_hemi(.40,5,1,
        #         (np.deg2rad(-90),np.deg2rad(90)),
        #         (np.deg2rad(45),np.deg2rad(45)),center_pos=table_center,look_pos=table_center))):
        #     # approximate relevant regions using downsampled LERF rendering
        #     c2w = ns_wrapper.visercam_to_ns(cam_pose.matrix[:3,:])
        #     rscam = RealsenseCamera.get_camera(c2w, downscale=1/5)
        #     outputs = ns_wrapper(camera=rscam, render_lerf=True)
        #     lerf_xyz.append(outputs['xyz'].reshape(-1, 3))
        #     lerf_relevancy.append(outputs[f'relevancy_{len(ns_wrapper.pipeline.image_encoder.positives)-1}'].flatten())
        # lerf_xyz = np.concatenate(lerf_xyz, axis=0)
        # lerf_relevancy = np.concatenate(lerf_relevancy, axis=0)
        # lerf_points_o3d = o3d.utility.Vector3dVector(lerf_xyz)

        center_pos_matrix = np.array([[ 1., 0., 0., 0.45], [0., -0.70710678,  0.70710678, -0.28284271],[ 0, -0.70710678, -0.70710678,  0.10284271]])
        c2w = ns_wrapper.visercam_to_ns(center_pos_matrix)
        rscam = RealsenseCamera.get_camera(c2w, downscale=1/5)
        # import pdb; pdb.set_trace()
        lerf_pcd, lerf_relevancy = ns_wrapper.get_lerf_pointcloud(rscam)
        lerf_pcd.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(lerf_pcd.points), ns_wrapper.applied_transform)) #nerfstudio pc to world/viser pc
        lerf_xyz = np.asarray(lerf_pcd.points)
        lerf_points_o3d = lerf_pcd.points
        # ^ These outputs are what the rest of the script needs

        # Visualize the relevancy pointcloud 
        # import pdb; pdb.set_trace()
        colors = lerf_relevancy.squeeze()
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        colors = matplotlib.colormaps['viridis'](colors)[:, :3]
        server.add_point_cloud(
            name=f"lerf_pointcloud",
            points=lerf_xyz,
            colors=colors,
            point_size=0.003,
        )
        best_grasp = None
        best_score = 0
        for ind,cam_pose in enumerate(_generate_hemi(hemi_radius.value,hemi_theta_N.value,hemi_phi_N.value,
                (np.deg2rad(-hemi_th_range.value),np.deg2rad(hemi_th_range.value)),
                (np.deg2rad(hemi_phi_down.value),np.deg2rad(hemi_phi_up.value)),center_pos=table_center,look_pos=table_center)):
            c2w = cam_pose.matrix[:3,:]
            start = time.time()
            rgbd_cropped = world_pointcloud.copy()
            rgbd_cropped.vertices = tr.transformations.transform_points(
                rgbd_cropped.vertices,
                np.linalg.inv(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
            )
            print("Transform time: ", time.time() - start)

            gg = graspnet_large(rgbd_cropped)
            gg.transform(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
            print(f"Grasp pred time: {time.time() - start:.2f}s")
            start = time.time()
            gg = gg[gg.translations[:, 2] > table_center[2]]

            # select grasps that point relatively upwards (up to 70 degrees from z-axis)
            rotation_angle = np.arccos(((-gg.rotation_matrices[:, :, 0]) * np.array([0, 0, 1])).sum(axis=1))
            gg = gg[np.where(rotation_angle < 80.0/180.0*np.pi)[0]]
            gg = gg.nms(translation_thresh=0.02, rotation_thresh=30.0/180.0*np.pi)
            # select grasps that are not too close to the table
            gg = local_collision_detection(gg)
            print(f"Collision detection time: {time.time() - start:.2f}s")

            print(f"Post proc time: {time.time() - start:.2f}s")
            

            if len(gg) > 0:
                # below line is for grasp score
                # print([gg[i].score for i in range(len(gg))])
                scores = []
                for i,grasp in enumerate(gg):
                    box = get_bbox_from_grasp(grasp)
                    # get the indices of the lerf_xyz pointcloud which lie inside the box and avg
                    pts = box.get_point_indices_within_bounding_box(lerf_points_o3d)
                    score = lerf_relevancy[pts].mean()
                    if score>best_score:
                        best_score = score
                        best_grasp = grasp
                    scores.append(score)

                #All visualization stuff
                scores = np.array(scores)
                scores /= lerf_relevancy.max()
                gg = gg.to_open3d_geometry_list()
                gg_tr = [
                    tr.Trimesh(vertices=np.asarray(gg[i].vertices), faces=np.asarray(gg[i].triangles))
                    for i in range(len(gg))
                ]
                color = matplotlib.colormaps['viridis'](scores)[:, :3]
                for i, gg in enumerate(gg_tr):
                    grasp_list.append(server.add_mesh(
                        name=f'view_{ind}/grasps_{i}',
                        vertices=gg.vertices,
                        faces=gg.faces,
                        color=color[i,:]
                        # below line is for grasp score
                        #color=np.array([255-min(2*scores[i], 255), min(255, 2*scores[i]), 0]).astype(np.uint8)
                    ))

        if best_grasp is not None:
            #Visualize the grasp
            gg=best_grasp.to_open3d_geometry()
            gg_tr = tr.Trimesh(vertices=np.asarray(gg.vertices), faces=np.asarray(gg.triangles))
            grasp_list.append(server.add_mesh(
                name=f'best grasp',
                vertices=gg_tr.vertices,
                faces=gg_tr.faces,
                color=np.array((0,0,255))
            ))
            #the conventions for graspnet and ur5 are different, so we need to rotate the grasp
            robot_frame_R = best_grasp.rotation_matrix @ RigidTransform.y_axis_rotation(np.pi/2)@RigidTransform.z_axis_rotation(np.pi/2)
            grasp_pose = RigidTransform(translation=best_grasp.translation,rotation=robot_frame_R)
            print("Best grasp score: ", best_score)
            print("Best grasp: ", grasp_pose)
            if 'y' in input("Grasp given pose?"):
                robot.move_joint(UR5_HOME_JOINT, vel=0.15)
                cur_q = robot.get_joints()
                PRE_T = RigidTransform(translation=[0, 0, -0.05],from_frame=grasp_pose.from_frame, to_frame=grasp_pose.from_frame)
                pre_grasp = grasp_pose * PRE_T
                traj, succ = motion_planner.get_trajectory(pre_grasp.matrix, cur_q)
                if succ:
                    robot.move_joint_path(traj,vels=[.2]*len(traj),accs=[1]*len(traj),blends = [0.01]*(len(traj)-1)+[0],asyn=False)
                    curp = robot.get_pose()
                    robot.move_pose(curp*PRE_T.inverse(),interp='tcp',vel=.15,acc=1)
                else:
                    print("No trajectory found")
            else:
                print("Not grasping")

        gen_grasp_button.disabled = False
        gen_grasp_text.disabled = False

    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)
