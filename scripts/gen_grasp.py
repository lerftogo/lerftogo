import viser
import viser.transforms as tf
import time
import numpy as np
import trimesh as tr
import tyro

import tqdm

from robot_lerf.graspnet_baseline.load_ns_model import NerfstudioWrapper, RealsenseCamera
from robot_lerf.graspnet_baseline.graspnet_module import GraspNetModule
from autolab_core import RigidTransform
import open3d as o3d
import matplotlib
from robot_lerf.graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector
from robot_lerf.capture_utils import _generate_hemi,HOME_POS


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

    # graspnet_large = GraspNetModule()
    # graspnet_large.init_net(graspnet_ckpt, cylinder_radius=0.04)

    # ns_wrapper = NerfstudioWrapper(config_path)
    # world_pointcloud,global_pointcloud,table_center = ns_wrapper.create_pointcloud()

    server = viser.ViserServer()
    server.add_frame(
        name="/world",
        axes_length=0.1,
        axes_radius=0.003,
        show_axes=True
    )

    # server.add_point_cloud(
    #     name=f"world_pointcloud",
    #     points=np.asarray(world_pointcloud.vertices),
    #     colors=np.asarray(world_pointcloud.visual.vertex_colors[:, :3]),
    #     point_size=0.002,
    # )
    gen_grasp_text = server.add_gui_text(
        name=f"LERF query",
        initial_value="",
    )
    gen_grasp_button = server.add_gui_button(
        name=f"Generate GRASPS",
    )

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
        gen_traj_slider.disabled = True

    with server.gui_folder("Hemisphere"):
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

    grasp_list, best_grasp_pose = [], None

    @gen_grasp_button.on_click
    def _(_):
        nonlocal grasp_list,world_pointcloud,global_pointcloud,best_grasp_pose
        gen_grasp_button.disabled = True
        gen_grasp_text.disabled = True
        lerf_word = gen_grasp_text.value.split(";")
        ns_wrapper.pipeline.image_encoder.set_positives(lerf_word)

        for grasp in grasp_list:
            grasp.remove()#GuiHandle

        # Get the LERF activation pointcloud for the given query
        # TODO convert this to a dino floodfill-style thing

        lerf_relevancy, lerf_xyz = [], []
        for ind,cam_pose in enumerate(tqdm.tqdm(_generate_hemi(.40,5,1,
                (np.deg2rad(-90),np.deg2rad(90)),
                (np.deg2rad(45),np.deg2rad(45)),center_pos=table_center,look_pos=table_center))):
            # approximate relevant regions using downsampled LERF rendering
            c2w = ns_wrapper.visercam_to_ns(cam_pose.matrix[:3,:])
            rscam = RealsenseCamera.get_camera(c2w, downscale=1/5)
            outputs = ns_wrapper(camera=rscam, render_lerf=True)
            lerf_xyz.append(outputs['xyz'].reshape(-1, 3))
            lerf_relevancy.append(outputs[f'relevancy_{len(ns_wrapper.pipeline.image_encoder.positives)-1}'].flatten())
        lerf_xyz = np.concatenate(lerf_xyz, axis=0)
        lerf_relevancy = np.concatenate(lerf_relevancy, axis=0)
        lerf_points_o3d = o3d.utility.Vector3dVector(lerf_xyz)

        # center_pos_matrix = np.array([[ 1., 0., 0., 0.45], [0., -0.70710678,  0.70710678, -0.28284271],[ 0, -0.70710678, -0.70710678,  0.10284271]])
        # c2w = ns_wrapper.visercam_to_ns(center_pos_matrix)
        # rscam = RealsenseCamera.get_camera(c2w, downscale=1/5)
        # # import pdb; pdb.set_trace()
        # lerf_pcd, lerf_relevancy = ns_wrapper.get_lerf_pointcloud(rscam)
        # lerf_pcd.points = o3d.utility.Vector3dVector(tr.transformations.transform_points(np.asarray(lerf_pcd.points), ns_wrapper.applied_transform)) #nerfstudio pc to world/viser pc
        # lerf_xyz = np.asarray(lerf_pcd.points)
        # lerf_points_o3d = lerf_pcd.points
        # # ^ These outputs are what the rest of the script needs

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
        gg_all = None
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
            mask = np.where(rotation_angle < 50.0/180.0*np.pi)[0]
            
            gg_ok = gg[mask]
            for grasp in gg[~mask]:
                rotation_angle = np.arccos(((-grasp.rotation_matrix[:, 0]) * np.array([0, 0, 1])).sum())
                rotation_axis = grasp.rotation_matrix[:, 1]
                rotmat_0 = tr.transformations.rotation_matrix(rotation_angle, rotation_axis)[:3, :3] @ grasp.rotation_matrix
                rotmat_1 = tr.transformations.rotation_matrix(-rotation_angle, rotation_axis)[:3, :3] @ grasp.rotation_matrix
                rotmat_0_angle = np.arccos(((-rotmat_0[:, 0]) * np.array([0, 0, 1])).sum())
                rotmat_1_angle = np.arccos(((-rotmat_1[:, 0]) * np.array([0, 0, 1])).sum())
                if rotmat_0_angle > 50.0/180.0*np.pi and rotmat_1_angle > 50.0/180.0*np.pi:
                    pass
                elif rotmat_0_angle < rotmat_1_angle:
                    rotmat = tr.transformations.rotation_matrix(rotation_angle-50.0/180.0*np.pi, rotation_axis)[:3, :3] @ grasp.rotation_matrix
                    grasp.rotation_matrix = rotmat.flatten()
                    gg_ok.add(grasp)
                else:
                    rotmat = tr.transformations.rotation_matrix(-rotation_angle+50.0/180.0*np.pi, rotation_axis)[:3, :3] @ grasp.rotation_matrix
                    grasp.rotation_matrix = rotmat.flatten()
                    gg_ok.add(grasp)
            gg = gg_ok
            
            gg = gg.nms(translation_thresh=0.02, rotation_thresh=30.0/180.0*np.pi)
            # select grasps that are not too close to the table
            gg = local_collision_detection(gg)
            print(f"Collision detection time: {time.time() - start:.2f}s")
            print(f"Post proc time: {time.time() - start:.2f}s")
            if gg_all is None:
                gg_all = gg
            else:
                gg_all.add(gg)
            
        if len(gg_all) > 0:
            # below line is for grasp score
            # print([gg[i].score for i in range(len(gg))])
            lerf_scores = []
            for i,grasp in enumerate(gg_all):
                box = get_bbox_from_grasp(grasp)
                # get the indices of the lerf_xyz pointcloud which lie inside the box and avg
                pts = box.get_point_indices_within_bounding_box(lerf_points_o3d)
                score = lerf_relevancy[pts].mean()
                if score>best_score:
                    best_score = score
                    best_grasp = grasp
                lerf_scores.append(score)

            #All visualization stuff
            lerf_scores = np.array(lerf_scores)
            lerf_scores /= lerf_relevancy.max()
            grasp_scores = np.array([g.score for g in gg_all])
            grasp_scores /= grasp_scores.max()

            gg_all = gg_all.to_open3d_geometry_list()
            gg_tr = [
                tr.Trimesh(vertices=np.asarray(gg_all[i].vertices), faces=np.asarray(gg_all[i].triangles))
                for i in range(len(gg_all))
            ]
            lerf_color = matplotlib.colormaps['viridis'](lerf_scores)[:, :3]
            grasp_color = matplotlib.colormaps['viridis'](grasp_scores)[:, :3]
            for i, gg in enumerate(gg_tr):
                foo = server.add_mesh(
                    name=f'lerf/grasps_{i}',
                    vertices=gg.vertices,
                    faces=gg.faces,
                    color=lerf_color[i,:],
                    clickable=True
                    # below line is for grasp score
                    #color=np.array([255-min(2*scores[i], 255), min(255, 2*scores[i]), 0]).astype(np.uint8)
                )
                @foo.on_click
                def _(_):
                    print("clicked grasp")
                grasp_list.append(foo)
                # grasp_list.append(server.add_mesh(
                #     name=f'grasp/grasps_{i}',
                #     vertices=gg.vertices,
                #     faces=gg.faces,
                #     color=grasp_color[i,:]
                #     # below line is for grasp score
                #     #color=np.array([255-min(2*scores[i], 255), min(255, 2*scores[i]), 0]).astype(np.uint8)
                # ))

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
            # grasp_pose = RigidTransform(translation=best_grasp.translation,rotation=best_grasp.rotation_matrix)
            print("Best grasp score: ", best_score)
            print("Best grasp: ", grasp_pose)
            best_grasp_pose = grasp_pose
            # if 'y' in input("Grasp given pose?"):
            #     grasp_pose.save("grasp_pose.tf")
            # else:
            #     print("Not grasping")

        gen_grasp_button.disabled = False
        gen_grasp_text.disabled = False

    @gen_traj_button.on_click
    def _(_):
        gen_traj_button.disabled = True

        from goto_grasp import moveto_grasp
        traj, get_robot = moveto_grasp(best_grasp_pose)
        if get_robot is None:
            print("Robot trajectory invalid")
            gen_traj_button.disabled = False
            return
        
        robot = get_robot(0)
        server.add_mesh(
            name="ur5",
            vertices=robot.vertices,
            faces=robot.faces,
        )
        
        gen_traj_button.disabled = False
        gen_traj_slider.disabled = False
        @gen_traj_slider.on_update
        def _(_):
            step = gen_traj_slider.value
            robot = get_robot(step)
            server.add_mesh(
                name="ur5",
                vertices=robot.vertices,
                faces=robot.faces,
            )

    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)
