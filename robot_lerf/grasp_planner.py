"""
Wrapper around ur5_* modules and graspnetAPI 
to plan grasps and execute them on the UR5 robot.
"""

import time
from pathlib import Path
import trimesh as tr
from typing import Tuple, Any, List, Dict
from functools import partial

import numpy as np
from autolab_core import RigidTransform

import tyro
import viser
import viser.transforms as tf
import open3d as o3d

# from ur5py.ur5 import UR5Robot
from robot_lerf.ur5_motion_planning import UR5MotionPlanning
from graspnetAPI import Grasp

import multiprocessing as mp
import queue
import time

# For visualization
import yourdfpy
UR5_HOME_JOINT=[
    -3.2835257689105433,
    -1.7120874563800257,
    1.6925301551818848,
    -1.7154505888568323,
    -1.6483410040484827,
    1.5793789625167847
    ]
ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint", 
    "wrist_1_joint",
    "wrist_2_joint", 
    "wrist_3_joint",
    "camera_joint"
]
import viser

class UR5GraspPlanner:
    def __init__(self, urdf_path: Path, root_name="ur5"):
        self.urdf = yourdfpy.URDF.load(
            urdf_path, 
            filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
            load_collision_meshes=True,
            build_collision_scene_graph=True,
            )
        self.motion_planner = UR5MotionPlanning()
        self._server = None
        self.root_name = root_name

    def create_robot(self, server: viser.ViserServer, root_transform=viser.SceneNodeHandle) -> None:
        # root_transform is from calling add_frame
        self._server = server
        self._ur5_frame = root_transform
        fingers = []
        coll_joints = []
        for frame_name, value in self.urdf.collision_scene.geometry.items():
            assert isinstance(value, tr.Trimesh)
            mesh_frame_name = self._frame_name_with_parents(frame_name) + "/mesh"
            self._server.add_mesh(
                f"/{self.root_name}/" + mesh_frame_name,
                vertices=value.vertices,
                faces=value.faces,
                color=(150, 150, 150),
            )
            if ("shape0" in frame_name) or ("geometry_13" in frame_name) or ("geometry_18" in frame_name): 
                fingers.append(frame_name)
            if ("camera" in frame_name) or ("wrist3" in frame_name) or ("wrist2" in frame_name):
                coll_joints.append(frame_name)
        self.coll_groups = [[fingers[1], fingers[2],fingers[3]], [fingers[6], fingers[7], fingers[8]], coll_joints]

    def _get_world_pos_from_name(self, frame_name: str) -> RigidTransform:
        """Quick [frame_name] -> [world]

        Args:
            frame_name (str): name of the frame (as defined in urdf)

        Returns:
            RigidTransform: transform from [frame_name] to [world]
        """
        transform = None
        frame_name = self.urdf.collision_scene.graph.transforms.parents[frame_name]
        while frame_name != self.urdf.collision_scene.graph.base_frame:
            parent = self.urdf.collision_scene.graph.transforms.parents[frame_name]
            T_parent_child = self.urdf.get_transform(frame_name, parent)
            T_parent_child = RigidTransform(
                rotation=T_parent_child[:3, :3],
                translation=T_parent_child[:3, 3],
                from_frame=frame_name,
                to_frame=parent,
            )
            if transform is None:
                transform = T_parent_child
            else:
                transform = T_parent_child * transform
            frame_name = parent

        transform = RigidTransform(
            rotation=tf.SO3(self._ur5_frame.wxyz).as_matrix(),
            translation=self._ur5_frame.position,
            from_frame="world",
            to_frame=frame_name,
        ) * transform
        
        return transform

    def get_coll_bbox_list(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get the oriented bounding box of the collision groups

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: list of transforms and extents of the collision groups
        """
        all_transforms, all_extents = [], []

        for group in self.coll_groups:
            # get the bounding box of the group
            group_mesh = []
            for item in group:
                bounds = self.urdf.collision_scene.geometry[item].bounds
                box = tr.creation.box(bounds[1]-bounds[0])
                box.vertices += (bounds[0] + bounds[1]) / 2
                box.vertices = tr.transformations.transform_points(box.vertices, self._get_world_pos_from_name(item).matrix)
                group_mesh.append(box)
            group_mesh = sum(group_mesh, tr.Trimesh())
            transform, extents = tr.bounds.oriented_bounds(group_mesh)
            all_transforms.append(np.linalg.inv(transform))
            all_extents.append(extents)

        return all_transforms, all_extents

    def _frame_name_with_parents(self, frame_name: str) -> str:
        frames = []
        while frame_name != self.urdf.collision_scene.graph.base_frame:
            frames.append(frame_name)
            frame_name = self.urdf.collision_scene.graph.transforms.parents[frame_name]
        return "/".join(frames[::-1])

    def goto_joints(self, joints: np.ndarray):
        assert self._server is not None
        cfg={ARM_JOINT_NAMES[i]:joints[i] for i in range(len(joints))}
        self.urdf.update_cfg(cfg)
        for joint in self.urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self.urdf.get_transform(joint.child, joint.parent)
            self._server.add_frame(
                f"/{self.root_name}/" + self._frame_name_with_parents(joint.child),
                wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                position=T_parent_child[:3, 3],
                show_axes=False,
            )

    def check_grasp_scene_collision(self, target_joint_angles: np.ndarray, world_pointcloud: tr.PointCloud) -> bool:
        """Check if the grasp pose will result in a collision with the scene.

        Args:
            grasp_pose (RigidTransform): _description_

        Returns:
            bool: True if no collision (i.e., success), False otherwise
        """
        self.goto_joints(target_joint_angles)

        world_points = o3d.utility.Vector3dVector(world_pointcloud.vertices)

        transforms, extents = self.get_coll_bbox_list()
        for trans, ext in zip(transforms, extents):
            coll_body = o3d.geometry.OrientedBoundingBox(trans[:3, 3], trans[:3, :3], ext)
            pts = coll_body.get_point_indices_within_bounding_box(world_points)
            if len(pts) > 0:
                print("THIS WILL COLLIDE!")
                return False
        
        print("NO COLLISION!")
        return True
        
    def create_traj_from_grasp(self, grasp_pose: RigidTransform, world_pointcloud: tr.PointCloud) -> Tuple[Any, Any]:
        """Given a grasp pose, create a trajectory to execute the grasp.
        Currently, performs the following:
         - Move to a pre-grasp position, first rotating the last three joints, then rotating the first three joints.
         The pre-grasp position is 10cm above the grasp position (in EE frame)
         - Move forwards 10cm into the grasp position.

        Args:
            grasp_pose (RigidTransform): grasp pose, in UR5 EE pose
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)

        Returns:
            Tuple[Any, Any]: (trajectory, success) if success is True, (None, False) otherwise
                             (trajectory is np.ndarray with shape (n_steps, n_joints))
        """
        cur_q = UR5_HOME_JOINT

        PRE_T = RigidTransform(translation=[0, 0, -0.05],from_frame=grasp_pose.from_frame, to_frame=grasp_pose.from_frame)
        pre_grasp = grasp_pose * PRE_T

        traj, succ = self.motion_planner.get_trajectory(pre_grasp.matrix, cur_q)  # traj is a numpy array of shape (n_steps, n_joints)

        # Check for collision
        succ = succ and self.check_grasp_scene_collision(traj[-1, :], world_pointcloud=world_pointcloud)

        if not succ:
            return None, False

        # Rotate the last two joints to the pregrasp position, then rotate the first four joints to the pregrasp position.
        traj_last_three_joints = traj.copy()
        traj_last_three_joints[:, :3] = cur_q[:3]
        traj_first_three_joints = traj.copy()
        traj_first_three_joints[:, 3:] = traj_last_three_joints[-1, 3:]

        traj_remaining, succ = self.motion_planner.get_trajectory(grasp_pose.matrix, traj_first_three_joints[-1], allow_180=False)
        traj = np.concatenate([traj_last_three_joints, traj_first_three_joints, traj_remaining], axis=0)

        # Check for collision
        succ = succ and self.check_grasp_scene_collision(traj[-1, :], world_pointcloud=world_pointcloud)

        if not succ:
            return None, False

        return traj, succ


# class CollisionChecker:

#     ARM_JOINT_NAMES = [
#         "shoulder_pan_joint",
#         "shoulder_lift_joint",
#         "elbow_joint", 
#         "wrist_1_joint",
#         "wrist_2_joint", 
#         "wrist_3_joint",
#         "camera_joint"
#     ]

#     ARM_LINK_NAMES = [
#         "base_link",
#         "shoulder_link",
#         "upper_arm_link",
#         "forearm_link",
#         "wrist_1_link",
#         "wrist_2_link",
#         "wrist_3_link",
#         "ee_link",
#         "left_gripper",
#         "right_gripper",
#         "camera_mount"
#     ]

#     SAFE_COLLISION_LINKS = [
#         ('ee_link', 'wrist_3_link'),
#         ('camera_mount', 'ee_link'), 
#         ('forearm_link', 'upper_arm_link'), 
#         ('camera_mount', 'wrist_3_link'), 
#         ('wrist_2_link', 'wrist_3_link'), 
#         ('forearm_link', 'wrist_1_link'),
#         ('camera_mount', 'left_gripper'),
#         ('camera_mount', 'right_gripper'),
#         # ("table","left_gripper"),
#         # ("table","right_gripper"),
#         ("base","base_link")
#     ]

#     def __init__(self, n_proc):
#         # self.robot = URDF.load("/home/lawrence/dmodo/ur5_go_old/ur5_pybullet/urdf/real_arm_w_camera.urdf")
#         # self.robot = URDF.load('/home/lawrence/robotlerf/ur5bc/ur5/real_arm_no_offset.urdf')
#         self.robot = URDF.load('/home/lawrence/robotlerf/robot_lerf/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf')
#         self._n_proc = n_proc
#         self.base_mesh = trimesh.creation.box(
#             extents=(0.78,0.56,0.34),
#             transform=[[1,0,0,-0.19],[0,1,0,0],[0,0,1,-0.07],[0,0,0,1]],
#         )
#         self.table_mesh = trimesh.creation.box(
#             bounds = [[-0.5,-0.5,-.3],[0.5,0.5,-0.18]]
#             # extents=(1,1,.1),
#             # transform=[[1,0,0,.45],[0,1,0,0],[0,0,1,-0.18-.05],[0,0,0,1]],
#         )
#         self.output_queue = mp.Queue()
#         self.coll_procs = []
#         for i in range(self._n_proc):
#             self.coll_procs.append(
#                 FCLProc(
#                     self.robot.links,
#                     self.ARM_LINK_NAMES,
#                     self.base_mesh,
#                     self.table_mesh,
#                     self.output_queue,
#                 )
#             )
#             self.coll_procs[-1].daemon = True
#             self.coll_procs[-1].start()

#     def in_collision(self, q):
#         num_q, num_j = q.shape
#         cfg = {self.ARM_JOINT_NAMES[i]: q[:, i] for i in range(num_j)}

#         link_poses = self.robot.link_fk_batch(cfgs=cfg, use_names=True)

#         for i in range(self._n_proc):
#             self.coll_procs[i].collides(
#                 link_poses,
#                 np.arange(
#                     i * num_q // self._n_proc,
#                     (i + 1) * num_q // self._n_proc,
#                 ),
#                 safe_collisions=self.SAFE_COLLISION_LINKS,
#             )

#         # collect computed iks
#         collides = False
#         for _ in range(self._n_proc):
#             collides |= self.output_queue.get(True)[0]
#             if collides:
#                 break

#         return collides

#     def vis_init(self):
#         vis3d.figure()
#         vis3d.mesh(self.base_mesh)
#         vis3d.show(asynch=True,animate=False)


#         self.node_list = {}
#         for link in self.robot.links:
#             if link.name in self.ARM_LINK_NAMES:
#                 # print("pose",link,pose,pose.shape)
#                 n = vis3d.mesh(link.collision_mesh)
#                 self.node_list[link.name]=n


#     def vis(self,q):
#         num_q, num_j = q.shape
#         cfg = {self.ARM_JOINT_NAMES[i]: q[:, i] for i in range(num_j)}
#         link_poses = self.robot.link_fk_batch(cfgs=cfg, use_names=True)

#         # print("link_poses",link_poses["base_link"].shape)
        
#         # self.robot.animate(cfg_trajectory=cfg)
#         # print("qshape",num_q,num_j)

#         # vis3d.figure()
#         # vis3d.mesh(self.base_mesh)
#         # vis3d.show(asynch=True,animate=False)


#         # node_list = {}
#         # for link, pose in link_poses.items():
#         #     if link in self.ARM_LINK_NAMES:
#         #         # print("pose",link,pose,pose.shape)
#         #         n = vis3d.mesh(self.robot.link_map[link].collision_mesh,T_mesh_world= pose[0])
#         #         node_list[link]=n
#         # import pdb;pdb.set_trace()
#         fps = 125.0
#         for i in range(num_q):
#             time.sleep(1.0 / fps)
#             for link, pose in link_poses.items():
#                 if link in self.ARM_LINK_NAMES:
#                    self.node_list[link]._matrix=pose[i]

        

# class FCLProc(mp.Process):
#     """
#     Used for finding collisions in parallel using FCL.
#     """

#     def __init__(self, urdf: yourdfpy.URDF, scene_mesh_list: Dict[str: tr.Trimesh], output_queue: mp.Queue):
#         """
#         Args:
#         output_queue: mp.Queue, the queue that all the output data
#             that is computed is added to.
#         """
#         super().__init__()
#         self.urdf = urdf
#         self.output_queue = output_queue
#         self.input_queue = mp.Queue()

#         self.arm_mgr = tr.collision.CollisionManager()
#         self.base_mgr = tr.collision.CollisionManager()
#         for mesh_name, mesh in scene_mesh_list.items():
#             self.base_mgr.add_object(mesh_name, mesh)


#     def _collides(self, link_poses, inds, safe_collisions):
#         """ computes collisions."""
#         collides = False
#         for i in inds:
#             for link, pose in link_poses.items():
#                 if link in self.arm_links:
#                     self.arm_mgr.set_transform(link, pose[i])
#             _, names = self.base_mgr.in_collision_other(
#                 self.arm_mgr, return_names=True
#             )
#             for name in names:
#                 if name not in safe_collisions:
#                     print(name)
#                     collides = True
#                     break
            
#             is_collision, names = self.arm_mgr.in_collision_internal(return_names=True)
#             for name in names:
#                 if name not in safe_collisions:
#                     print(name)
#                     collides = True
#                     break
#         return collides

#     def run(self):
#         """
#         the main function of each FCL process.
#         """
#         for link in self.links:
#             if link.name in self.arm_links:
#                 self.arm_mgr.add_object(link.name, link.collision_mesh)

#         while True:
#             try:
#                 request = self.input_queue.get(timeout=1)
#             except queue.Empty:
#                 continue
#             if request[0] == "collides":
#                 self.output_queue.put((self._collides(*request[1:]),))

#     def collides(self, link_poses, inds, pind=None, safe_collisions=[]):
#         self.input_queue.put(("collides", link_poses, inds, safe_collisions))