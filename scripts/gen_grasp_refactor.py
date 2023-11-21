import time
import numpy as np
import trimesh as tr
import tyro

import os.path as osp
from pathlib import Path
import tqdm
import open3d as o3d
import matplotlib
from typing import List, Dict
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import viser
import viser.transforms as tf
from autolab_core import RigidTransform

from graspnetAPI import GraspGroup, Grasp
import yourdfpy
import itertools

from robot_lerf.graspnet_baseline.load_ns_model import NerfstudioWrapper, RealsenseCamera
from robot_lerf.graspnet_baseline.graspnet_module import GraspNetModule
from robot_lerf.graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector
from robot_lerf.capture_utils import _generate_hemi
from robot_lerf.grasp_planner_cmk import UR5GraspPlanner # , UR5_HOME_JOINT, ARM_JOINT_NAMES

import capture as lerf_capture


def get_relevancy_pointcloud(ns_wrapper: NerfstudioWrapper, **kwargs):
    """Get relevancy pointcloud, used to get semantic score

    Args:
        ns_wrapper (NerfstudioWrapper): nerf scene

    Returns:
        o3d.utility.Vector3DVector: points in pointcloud (xyz)
        np.ndarray: relevancy score
    """
    lerf_xyz, lerf_relevancy = [], []
    # for cam_pose in tqdm.tqdm(
    #     _generate_hemi(
    #         .40,5,1,
    #         (np.deg2rad(-90),np.deg2rad(90)),
    #         (np.deg2rad(45),np.deg2rad(45)),
    #         center_pos=kwargs['table_center'],
    #         look_pos=kwargs['table_center']
    #         )):
    #     # approximate relevant regions using downsampled LERF rendering
    #     c2w = ns_wrapper.visercam_to_ns(cam_pose.matrix[:3,:])
    #     rscam = RealsenseCamera.get_camera(c2w, downscale=1/5)
    #     outputs = ns_wrapper(camera=rscam, render_lerf=True)
    #     lerf_xyz.append(outputs['xyz'].reshape(-1, 3))
    #     num_pos = len(ns_wrapper.pipeline.image_encoder.positives)
    #     lerf_relevancy.append(outputs[f"relevancy_{num_pos-1}"].flatten())
    # lerf_xyz = np.concatenate(lerf_xyz, axis=0)
    # lerf_relevancy = np.concatenate(lerf_relevancy, axis=0)
    # lerf_points_o3d = o3d.utility.Vector3dVector(lerf_xyz)
    # return lerf_points_o3d, lerf_relevancy

    center_pos_matrix = np.array([[ 1., 0., 0., 0.45], [0., -0.70710678,  0.70710678, -0.28284271],[ 0, -0.70710678, -0.70710678,  0.10284271]])
    c2w = ns_wrapper.visercam_to_ns(center_pos_matrix)
    rscam = RealsenseCamera.get_camera(c2w, downscale=1/5)
    lerf_pcd, lerf_relevancy = ns_wrapper.get_lerf_pointcloud(rscam)
    if lerf_pcd is None:
        return None, None
    lerf_pcd.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(lerf_pcd.points), ns_wrapper.applied_transform)) #nerfstudio pc to world/viser pc
    lerf_xyz = np.asarray(lerf_pcd.points)
    lerf_points_o3d = lerf_pcd.points

    return lerf_points_o3d, lerf_relevancy

def get_grasps(
    graspnet: GraspNetModule,
    world_pointcloud: tr.PointCloud,
    hemisphere: List[RigidTransform],
    graspnet_batch_size: int = 20,
    # cam_pc_list: List[np.ndarray] = None,
    ) -> GraspGroup:
    """Get grasps from graspnet, as images taken from the hemisphere
    
    Args: 
        graspnet (GraspNetModule): graspnet module
        world_pointcloud (tr.PointCloud): world pointcloud
        hemisphere (List[RigidTransform]): list of camera poses
        cam_pc_list (List[np.ndarray]): list of camera pointclouds (each from the camera poses in `hemisphere`)
    
    Returns:
        GraspGroup: grasps
    """
    gg_all = None
    torch.cuda.empty_cache()
    # for c2w_list in itertools.islice(hemisphere, 0, len(hemisphere), graspnet_batch_size):
    for i in range(0, len(hemisphere), graspnet_batch_size):
        start = time.time()
        ind_range = range(i, min(i+graspnet_batch_size, len(hemisphere)))
        rgbd_cropped_list = []
        # if isinstance(c2w_list, RigidTransform):
        #     c2w_list = [c2w_list]
        # for c2w in c2w_list:
        for j in ind_range:
            # # if cam_pc_list is not None:
            # #     rgbd_cropped_list.append(cam_pc_list[j])
            # else:
            c2w = hemisphere[j].matrix[:3,:]
            rgbd_cropped = world_pointcloud.copy()
            rgbd_cropped.vertices = tr.transformations.transform_points(
                rgbd_cropped.vertices,
                np.linalg.inv(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
            )
            rgbd_cropped_list.append(rgbd_cropped)
        print("Transform time: ", time.time() - start)

        gg_list = graspnet(rgbd_cropped_list)
        for g_ind, gg in enumerate(gg_list):
            c2w = hemisphere[i + g_ind].matrix[:3,:]
            gg.transform(np.concatenate([c2w, [[0, 0, 0, 1]]], axis=0))
        print(f"Grasp pred time: {time.time() - start:.2f}s")
        start = time.time()

        gg_all_curr = gg_list[0]
        for gg in gg_list[1:]:
            gg_all_curr.add(gg)
        gg = gg_all_curr

        # If the grasps are too close to the ground, then lift them a bit.
        # This is hardcoded though, so it might not work for all scenes
        gg_translations = gg.translations
        gg_translations[gg_translations[:, 2] < -0.14] += np.tile(np.array([0, 0, 0.01]), ((gg_translations[:, 2] < -0.14).sum(), 1))
        gg.translations = gg_translations
        # gg[gg.translations[:, 2] < -0.16].translations += np.tile(np.array([0, 0, 0.04]), ((gg.translations[:, 2] < -0.16).sum(), 1))
        gg = gg[(gg.translations[:, 0] > 0.22) & (gg.translations[:, 2] < 0.05)]

        gg = gg[np.abs(gg.rotation_matrices[:, :, 1][:, 2]) < 0.5]

        # gg = gg[gg.scores > 0.6]
        if len(gg) == 0:
            continue

        gg = gg.nms(translation_thresh=0.05, rotation_thresh=30.0/180.0*np.pi)

        # select grasps that are not too close to the table
        # Currently, this function does general grasp filtering (using collision detection, grasp includes non-table components, ...)
        gg = graspnet.local_collision_detection(gg)

        print(f"Collision detection time: {time.time() - start:.2f}s")
        print(f"Post proc time: {time.time() - start:.2f}s")
        if gg_all is None:
            gg_all = gg
        else:
            gg_all.add(gg)

    if gg_all is None:
        return GraspGroup()
    
    gg_all = gg_all.nms(translation_thresh=0.05, rotation_thresh=30.0/180.0*np.pi)
    gg_all.sort_by_score()
    torch.cuda.empty_cache()

    return gg_all

def main(
    config_path: str,  # Nerfstudio model config path, of format outputs/.../config.yml
    graspnet_ckpt: str = 'robot_lerf/graspnet_baseline/logs/log_kn/checkpoint.tar',  # GraspNet checkpoint path
    urdf_path: str = "pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf"
    ):

    ns_wrapper = NerfstudioWrapper(config_path)
    world_pointcloud, global_pointcloud, table_center = ns_wrapper.create_pointcloud()

    grasp_planner = UR5GraspPlanner(Path(urdf_path))

    graspnet = GraspNetModule()
    graspnet.init_net(graspnet_ckpt, global_pointcloud, cylinder_radius=0.04, floor_height=grasp_planner.FLOOR_HEIGHT)

    server = viser.ViserServer()
    server.add_frame(
        name="/world",
        axes_length=0.05,
        axes_radius=0.01,
        show_axes=True
    )

    server.add_point_cloud(
        name=f"/world_pointcloud",
        points=np.asarray(world_pointcloud.vertices),
        colors=np.asarray(world_pointcloud.visual.vertex_colors[:, :3]),
        point_size=0.002,
    )

    server.add_point_cloud(
        name=f"/coll_pointcloud",
        points=graspnet.pointcloud_vertices,
        colors=np.repeat(np.array([[0, 1, 0]]), len(graspnet.pointcloud_vertices), axis=0),
        point_size=0.002,
        visible=False
    )

    ur5_frame = server.add_frame(
        name=f"/ur5",
        wxyz=tf.SO3.from_z_radians(np.pi).wxyz,
        show_axes=False
    )
    grasp_planner.create_robot(server, root_transform=ur5_frame, use_visual=True)
    grasp_planner.goto_joints(grasp_planner.UR5_HOME_JOINT)

    gen_grasp_text = server.add_gui_text(
        name=f"LERF query",
        initial_value="",
    )
    gen_grasp_text_button = server.add_gui_button(
        name=f"LERF query generate",
    )

    with server.gui_folder("Capture/Train LERF"):
        lerf_train_button = server.add_gui_button(
            name=f"Train LERF",
        )
        lerf_load_button = server.add_gui_button(
            name=f"Load LERF",
        )
        lerf_dataset_path = server.add_gui_text(
            name=f"savedir",
            initial_value="",
        )

    """
    Grasp generation GUI
    """
    with server.gui_folder("Grasp generation"):
        gen_grasp_button = server.add_gui_button(
            name=f"Generate GRASPS",
        )
        reset_grasp_button = server.add_gui_button(
            name=f"Reset GRASPS",
        )

        """
        Set hemisphere parameters -- these are used for generating grasps
        """
        with server.gui_folder("Hemisphere"):
            hemi_radius = server.add_gui_number(
                name=f"radius",
                initial_value=2,
            )
            hemi_theta_N = server.add_gui_number(
                name=f"theta_N",
                initial_value=15,
            )
            hemi_phi_N = server.add_gui_number(
                name=f"phi_N",
                initial_value=10,
            )
            hemi_th_range = server.add_gui_number(
                name=f"th_bounds",
                initial_value=180,
            )
            hemi_phi_down = server.add_gui_number(
                name=f"phi_down",
                initial_value=0,
            )
            hemi_phi_up = server.add_gui_number(
                name=f"phi_up",
                initial_value=70,
            )
        hemi_buttons_list = [hemi_radius, hemi_theta_N, hemi_phi_N, hemi_th_range, hemi_phi_down, hemi_phi_up]


    with server.gui_folder("Trajectory"):
        gen_traj_button = server.add_gui_button(
            name=f"Generate TRAJECTORY",
        )
        gen_traj_slider = server.add_gui_slider(
            name=f"Step",
            min=0.0,
            max=1.0,
            initial_value=0.0,
            step=0.01
        )

    grasps, grasps_dict, lerf_scores = None, {}, []
    lerf_points_o3d, lerf_relevancy = None, None
    traj_grasp_ind = -1
    traj = None

    def add_grasps(grasps: GraspGroup, score: np.ndarray, score_threshold: float):
        nonlocal grasps_dict

        # clear the grasps first...
        for grasps_list in grasps_dict.values():
            for grasp in grasps_list:
                grasp.remove()
        grasps_dict = {}

        colormap = matplotlib.colormaps['RdYlGn']
        robot_frame_R = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(np.pi/2) @ RigidTransform.z_axis_rotation(np.pi/2)
        )

        # for each grasp, add it to the dictionary
        def curry_grasp(grasp):
            default_grasp = Grasp()
            default_grasp.depth = grasp.depth
            default_grasp.width = grasp.width
            default_grasp.height = grasp.height
            default_grasp = default_grasp.to_open3d_geometry()
            score = grasp.score/1.1
            frame_handle = server.add_frame(
                name=f'/lerf/grasps_{ind}',
                wxyz=tf.SO3.from_matrix(grasp.rotation_matrix).wxyz,
                position=grasp.translation,
                show_axes=False
            )
            frame_show_handle = server.add_frame(
                name=f'/lerf/grasps_{ind}/axes',
                axes_length=0.05,
                axes_radius=0.002,
                show_axes=True,
                visible=False
            )
            grasp_handle = server.add_mesh(
                name=f'/lerf/grasps_{ind}/mesh',
                vertices=np.asarray(default_grasp.vertices),
                faces=np.asarray(default_grasp.triangles),
                color=colormap(score)[:3],
            )
            ur5_handle = server.add_frame(
                name=f'/lerf/grasps_{ind}/ur5',
                wxyz=robot_frame_R.quaternion,
                position=np.array([0.02, 0, 0]),
                axes_length=0.05,
                axes_radius=0.002,
                show_axes=True,
                visible=False
            )
            return frame_handle, frame_show_handle, grasp_handle, ur5_handle

        grasps_selected = [grasp for (ind, grasp) in enumerate(grasps) if score[ind] > score_threshold]
        inds_selected = [ind for (ind, grasp) in enumerate(grasps) if score[ind] > score_threshold]

        grasps_selected = GraspGroup(np.stack([grasp.grasp_array for grasp in grasps_selected]))
        grasps_selected = grasps_selected.nms(translation_thresh=0.05, rotation_thresh=30.0/180.0*np.pi)
        grasps_selected = grasps_selected.sort_by_score()

        for ind, grasp in zip(inds_selected, grasps_selected):
            grasps_dict[ind] = curry_grasp(grasp)

    """
    Generate LERF pointcloud -- modifies:
        - lerf_points_o3d
        - lerf_relevancy
        - grasps_dict
        - lerf_scores (if grasps is not None)
        - traj_grasp_ind (if grasps is not None)
    """
    # @gen_grasp_text.on_update
    @gen_grasp_text_button.on_click
    def _(_):
        nonlocal lerf_points_o3d, lerf_relevancy, grasps_dict, lerf_scores, traj_grasp_ind
        gen_grasp_text_button.disabled = True
        gen_grasp_text.disabled = True
        gen_grasp_button.disabled = True
        reset_grasp_button.disabled = True

        lerf_word = gen_grasp_text.value.split(";")
        ns_wrapper.pipeline.image_encoder.set_positives(lerf_word)

        # Get the LERF activation pointcloud for the given query
        lerf_points_o3d, lerf_relevancy = get_relevancy_pointcloud(ns_wrapper, table_center=table_center)
        if lerf_points_o3d is None:
            gen_grasp_text_button.disabled = False
            gen_grasp_text.disabled = False
            gen_grasp_button.disabled = False
            reset_grasp_button.disabled = False
            return
        # Visualize the relevancy pointcloud 
        colors = lerf_relevancy.squeeze()
        colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-6)
        colors = matplotlib.colormaps['viridis'](colors)[:, :3]
        server.add_point_cloud(
            name=f"/lerf_pointcloud",
            points=np.asarray(lerf_points_o3d),
            colors=colors,
            point_size=0.003,
        )

        if grasps is not None:
            lerf_scores = []
            for i, grasp in enumerate(grasps):
                box = graspnet.get_bbox_from_grasp(grasp)
                # get the indices of the lerf_xyz pointcloud which lie inside the box and avg
                pts = box.get_point_indices_within_bounding_box(lerf_points_o3d)
                if len(pts) == 0:
                    score = 0
                else:
                    score = lerf_relevancy[pts].mean()
                lerf_scores.append(score)

            #All visualization stuff
            lerf_scores = np.array(lerf_scores)
            lerf_scores /= lerf_relevancy.max()

            traj_grasp_thresh = np.quantile(lerf_scores, 0.9)
            add_grasps(grasps, lerf_scores, traj_grasp_thresh)

        gen_grasp_text.disabled = False
        gen_grasp_button.disabled = False
        gen_grasp_text_button.disabled = False
        reset_grasp_button.disabled = False

    """
    Removes all grasps from the scene -- modifies:
        - grasps_dict 
        - grasps
    """
    @reset_grasp_button.on_click
    def _(_):
        nonlocal grasps_dict, grasps
        for grasps_list in grasps_dict.values():
            for grasp in grasps_list:
                grasp.remove()
        grasps_dict = {}
        grasps = None

    """
    Generate grasps and their scores -- modifies:
        - grasps: GraspGroup
        - grasps_dict: Dict[int, List[GuiHandle]]
        - traj_grasp_ind: int (index of the grasp used for trajectory generation)
    `traj_grasp` is the grasp used for trajectory generation, and 
      can be changed by clicking on a different grasp.

    Frame structure:
        /lerf
        /lerf/grasps_0/
        /lerf/grasps_0/mesh      # Mesh of the grasp
        /lerf/grasps_0/ur5       # UR5's EE frame corresponding to the grasp
        ...
    """
    @gen_grasp_button.on_click
    def _(_):
        nonlocal grasps, grasps_dict, traj_grasp_ind

        gen_grasp_button.disabled = True
        gen_grasp_text.disabled = True
        reset_grasp_button.disabled = True
        for hemi_button in hemi_buttons_list:
            hemi_button.disabled = True

        lerf_word = gen_grasp_text.value.split(";")
        ns_wrapper.pipeline.image_encoder.set_positives(lerf_word)

        for grasps_list in grasps_dict.values():
            for grasp in grasps_list:
                grasp.remove()
        grasps_dict = {}

        grasp_hemisphere = _generate_hemi(
            hemi_radius.value,hemi_theta_N.value,hemi_phi_N.value,
            (np.deg2rad(-hemi_th_range.value),np.deg2rad(hemi_th_range.value)),
            (np.deg2rad(hemi_phi_down.value),np.deg2rad(hemi_phi_up.value)),
            center_pos=table_center,look_pos=table_center
            )
        grasps = get_grasps(graspnet, world_pointcloud, grasp_hemisphere)

        if len(grasps) < 0:
            return
        
        scores = grasps.scores
        score_threshold = np.quantile(scores, 0.5)
        add_grasps(grasps, scores, score_threshold)
        
        gen_grasp_button.disabled = False
        gen_grasp_text.disabled = False
        reset_grasp_button.disabled = False
        for hemi_button in hemi_buttons_list:
            hemi_button.disabled = False

    # """
    # Reset the up-direction of the viewer, doesn't modify anything.
    # """
    # gui_reset_up = server.add_gui_button("Reset up direction")
    # @gui_reset_up.on_click
    # def _(_) -> None:
    #     for client in server.get_clients().values():
    #         client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
    #             [0.0, -1.0, 0.0]
    #         ) 

    """
    Generate a trajectory to the selected grasp -- modifies:
        - traj
    """
    @gen_traj_button.on_click
    def _(_):
        nonlocal traj
        if grasps is None:
            return
        
        if lerf_scores is None or len(lerf_scores) == 0:
            scores = grasps.scores
        else:
            scores = lerf_scores

        gen_traj_button.disabled = True

        # traj_grasp_ind_list = np.argsort(grasps.scores)[::-1]
        traj_grasp_ind_list = np.argsort(scores)[::-1]
        for traj_grasp_ind in traj_grasp_ind_list:
            ur52grasp_pose = grasps_dict[traj_grasp_ind][-1]
            grasp2world_pose = grasps_dict[traj_grasp_ind][0]

            num_rotations_test = 8

            succ_traj_list = [] # store (traj, fin_pose)
            ur5_frame.visible = False

            start = time.time()
            print("Trying grasp", traj_grasp_ind)
            for i in range(num_rotations_test):
                print("Trying rotation", i)
                grasp_pose = RigidTransform(
                    translation=grasp2world_pose.position,
                    rotation=tf.SO3(grasp2world_pose.wxyz).as_matrix(),
                    from_frame="grasp",
                    to_frame="world"
                ) * RigidTransform(
                    rotation=RigidTransform.y_axis_rotation(i * (2*np.pi)/num_rotations_test),
                    from_frame="grasp",
                    to_frame="grasp"
                ) * RigidTransform(
                    translation=ur52grasp_pose.position,
                    rotation=tf.SO3(ur52grasp_pose.wxyz).as_matrix(),
                    from_frame="grasp/ee",
                    to_frame="grasp"
                )
                if grasp_pose.matrix[:, 2][2] > 0:
                    continue
                
                traj, succ, fin_pose = grasp_planner.create_traj_from_grasp(grasp_pose, world_pointcloud=world_pointcloud)
                if succ:
                    print(" - Success")
                    succ_traj_list.append((traj, fin_pose))
                else:
                    print(" - Failed")

            # if not succ:
            if len(succ_traj_list) == 0:
                print("None succeeded")
            else:
                print("Succeeded")
                break
            
            # traj = None
            # grasp_planner.goto_joints(grasp_planner.UR5_HOME_JOINT)
            # gen_traj_slider.value = 0.0
        # else:
            # find the best traj!

        if len(succ_traj_list) == 0:
            print("No trajectory found")
            traj = None
            gen_traj_slider.value = 0.0
            gen_traj_button.disabled = False
            return

        min_dist = np.inf
        best_traj, best_end_pose = None, None
        for curr_traj, end_pose in succ_traj_list:
            dist = np.linalg.norm(curr_traj[0, :] - curr_traj[-1, :])
            # dist matters, but making sure that the end pose is aligned is very important.
            if dist < min_dist and ((best_end_pose is None) or (np.linalg.norm(best_end_pose.translation-grasp_pose.translation) > np.linalg.norm(end_pose.translation-grasp_pose.translation))):
                min_dist = dist
                best_traj = curr_traj
                best_end_pose = end_pose

        traj, fin_pose = best_traj, best_end_pose
        succ = True

        traj_up, succ_up = grasp_planner.create_traj_lift_up(
            traj[-1, :],
            fin_pose,
            0.2,
            world_pointcloud=world_pointcloud,
        )
        succ = succ and succ_up

        if not succ:
            print("Failed for the lift")
            traj = None
            grasp_planner.goto_joints(grasp_planner.UR5_HOME_JOINT)
            gen_traj_slider.value = 0.0
        else:
            print("succeeded!")
            traj = np.concatenate([traj, traj_up], axis=0)
            gen_traj_slider.value = 0.75

        print(f"Time taken: {time.time() - start}")

        # reset ur5 pose
        ur5_frame.visible = True
        gen_traj_button.disabled = False

    """
    Update the UR5 robot vis based on the trajectory
    """
    @gen_traj_slider.on_update
    def _(_):
        if traj is None:
            return
        curr_joint = traj[int(gen_traj_slider.value*(len(traj)-1)), :]
        grasp_planner.goto_joints(curr_joint)

    """
    Move the UR5 robot to the trajectory!
    Be careful with this one, it will actually move the robot!
    A checkbox exists to disable this, and you also need to confirm with "y".
    """
    with server.gui_folder("Robot"):
        lerf_capture_button = server.add_gui_button(
            name=f"Capture LERF",
        )
        robot_button = server.add_gui_button(
            name="MOVE ROBOT TO trajectory"
        )
        robot_checkbox = server.add_gui_checkbox(
            name="Use Robot",
            initial_value=False
        )
        
        robot = None
        @robot_checkbox.on_update
        def _(_):
            nonlocal robot
            if not robot_checkbox.value:
                return
            if robot is not None: 
                return
            from ur5py.ur5 import UR5Robot
            robot = UR5Robot(gripper=True)
            robot.set_tcp(RigidTransform(rotation=np.eye(3),translation=np.array([0,0,0.16])))
            time.sleep(0.5)
            robot.move_joint(grasp_planner.UR5_HOME_JOINT)
            time.sleep(1)

        @lerf_capture_button.on_click
        def _(_):
            nonlocal robot
            if (robot is None) or (not robot_checkbox.value):
                return
            if lerf_dataset_path.value == "":
                print("Please enter a valid path")
                return
            lerf_capture_button.disabled = True
            lerf_train_button.disabled = True
            # this internally updates the TCP pose, so we need to reset it when we're done.
            lerf_capture.main(lerf_dataset_path.value, rob=robot)
            time.sleep(0.5)
            robot.set_tcp(RigidTransform(rotation=np.eye(3),translation=np.array([0,0,0.16])))
            time.sleep(0.5)

        @robot_button.on_click
        def _(_):
            nonlocal robot
            if traj is None:
                return
            if (robot is None) or (not robot_checkbox.value):
                return
            if 'y' in input("Go to grasp pose?"):
                robot.move_joint(grasp_planner.UR5_HOME_JOINT, asyn=False)
                time.sleep(0.5)
                robot.gripper.open()
                traj_goto, traj_lift = traj[:60], traj[60:]
                robot.move_joint_path(
                    traj_goto,
                    vels=[.2]*len(traj_goto),
                    accs=[1]*len(traj_goto),
                    blends = [0.02]*(len(traj_goto)-1)+[0],
                    asyn=False
                    )
                if 'y' in input("Close gripper?"):
                    robot.gripper.close()
                    time.sleep(0.5)
                    if 'y' in input("Lift gripper"):
                        robot.move_joint_path(
                            traj_lift,
                            vels=[.2]*len(traj_lift),
                            accs=[1]*len(traj_lift),
                            blends = [0.02]*(len(traj_lift)-1)+[0],
                            asyn=False
                            )
                        time.sleep(0.5)
                        if 'y' in input("Open gripper"):
                            robot.gripper.open()
                            time.sleep(0.5)
                            print("done")

    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)


            # # select grasps that point relatively upwards (up to 70 degrees from z-axis)
            # rotation_angle = np.arccos(((-gg.rotation_matrices[:, :, 0]) * np.array([0, 0, 1])).sum(axis=1))
            # mask = np.where(rotation_angle < 50.0/180.0*np.pi)[0]
            
            # gg_ok = gg[mask]
            # for grasp in gg[~mask]:
            #     rotation_angle = np.arccos(((-grasp.rotation_matrix[:, 0]) * np.array([0, 0, 1])).sum())
            #     rotation_axis = grasp.rotation_matrix[:, 1]
            #     rotmat_0 = tr.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3] @ grasp.rotation_matrix
            #     rotmat_1 = tr.transformations.rotation_matrix(-rotation_angle, rotation_axis)[:3, :3] @ grasp.rotation_matrix
            #     rotmat_0_angle = np.arccos(((-rotmat_0[:, 0]) * np.array([0, 0, 1])).sum())
            #     rotmat_1_angle = np.arccos(((-rotmat_1[:, 0]) * np.array([0, 0, 1])).sum())
            #     if rotmat_0_angle > 50.0/180.0*np.pi and rotmat_1_angle > 50.0/180.0*np.pi:
            #         pass
            #     elif rotmat_0_angle < rotmat_1_angle:
            #         rotmat = tr.transformations.rotation_matrix(rotation_angle-50.0/180.0*np.pi, rotation_axis)[:3, :3] @ grasp.rotation_matrix
            #         grasp.rotation_matrix = rotmat.flatten()
            #         gg_ok.add(grasp)
            #     else:
            #         rotmat = tr.transformations.rotation_matrix(-rotation_angle+50.0/180.0*np.pi, rotation_axis)[:3, :3] @ grasp.rotation_matrix
            #         grasp.rotation_matrix = rotmat.flatten()
            #         gg_ok.add(grasp)
            # gg = gg_ok
            

                    # robot_frame_R = grasp.rotation_matrix @ RigidTransform.y_axis_rotation(np.pi/2) @ RigidTransform.z_axis_rotation(np.pi/2)
                    # grasp_pose = RigidTransform(translation=grasp.translation,rotation=robot_frame_R)
                    # traj_grasp = grasp_pose