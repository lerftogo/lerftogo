import time
from pathlib import Path
import trimesh as tr

import numpy as np
from autolab_core import RigidTransform

import tyro
import viser
import viser.transforms as tf

from graspnetAPI import Grasp

from robot_lerf.grasp_planner_cmk import UR5GraspPlanner

def main(
    urdf_path: str = "pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf",
    grasp_pose_tf: str = "grasp_pose.tf"
    ):

    grasp_planner = UR5GraspPlanner(Path(urdf_path))
    world_pointcloud = tr.load("world_pointcloud.ply")

    server = viser.ViserServer()
    server.add_frame(
        name="/world",
        axes_length=0.05,
        axes_radius=0.003,
        show_axes=True
    )

    server.add_point_cloud(
        name=f"/world_pointcloud",
        points=np.asarray(world_pointcloud.vertices),
        colors=np.asarray(world_pointcloud.visual.vertex_colors[:, :3]),
        point_size=0.002,
    )

    ur5_frame = server.add_frame(
        name=f"/ur5",
        wxyz=tf.SO3.from_z_radians(np.pi).wxyz,
        position=np.array([0, 0, 0]),
        show_axes=False
    )
    grasp_planner.create_robot(server, ur5_frame)
    grasp_planner.goto_joints(grasp_planner.UR5_HOME_JOINT, show_collbodies=True)

    grasp_depth = 0.06
    # grasp_pose = RigidTransform.load(grasp_pose_tf)
    robot_frame_R = RigidTransform(
        rotation=RigidTransform.y_axis_rotation(np.pi/2) @ RigidTransform.z_axis_rotation(np.pi/2)
    )

    grasp_controls = server.add_transform_controls(
        name="/grasp",
        scale=0.1,
    )
    # grasp_controls.wxyz = grasp_pose.quaternion
    # grasp_controls.position = grasp_pose.translation
    gn_frame = server.add_frame(
        name="/grasp/gn",
        wxyz=robot_frame_R.quaternion,
        position=np.array([grasp_depth-0.02, 0, 0]),
        axes_length=0.05,
        axes_radius=0.003,
        show_axes=False
    )
    grasp_gna = Grasp()
    grasp_mesh = grasp_gna.to_open3d_geometry()
    server.add_mesh(
        name="/grasp/mesh",
        vertices=np.asarray(grasp_mesh.vertices),
        faces=np.asarray(grasp_mesh.triangles),
    )

    """
    All the functions that affect trajectory selection -- includes:
     - Trajectory generation
    """
    traj = None
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

        @gen_traj_button.on_click
        def _(_):
            nonlocal traj
            grasp_pose = RigidTransform(
                translation=grasp_controls.position,
                rotation=tf.SO3(grasp_controls.wxyz).as_matrix(),
                from_frame="grasp",
                to_frame="world"
            ) * RigidTransform(
                translation=gn_frame.position,
                rotation=tf.SO3(gn_frame.wxyz).as_matrix(),
                from_frame="grasp/gn",
                to_frame="grasp"
            )

            gen_traj_button.disabled = True

            start = time.time()
            traj, succ, fin_pose = grasp_planner.create_traj_from_grasp(grasp_pose, world_pointcloud=world_pointcloud)
                
            if not succ:
                print("No trajectory found")
                traj = None
                gen_traj_slider.value = 0.0
                gen_traj_button.disabled = False
                return

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
            # ur5_frame.visible = True
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

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    tyro.cli(main)
