"""
Wrapper around ur5_* modules and graspnetAPI 
to plan grasps and execute them on the UR5 robot.

Note:
    - This process tries to avoid multiprocessing -- collision geometries are sufficiently simple. 
    - Most of the collision logic is resolved using yourdfpy scene graph.
"""

import time
from pathlib import Path
from typing import Tuple, Any, List, Dict
from functools import partial

import numpy as np
from autolab_core import RigidTransform

import trimesh as tr
import open3d as o3d

import tyro
import viser
import viser.transforms as tf

import yourdfpy

from graspnetAPI import GraspGroup
from robot_lerf.ur5_robot import UR5RobotKinematics
from robot_lerf.graspnet_baseline.graspnet_module import GraspNetModule


class UR5GraspPlanner:
    UR5_HOME_JOINT = [
        -3.280724589024679,
        -1.711092774068014,
        1.6831812858581543, 
        -1.5566170851336878, 
        -1.5222366491900843, 
        1.418628215789795
        ]

    # UR5_HOME_JOINT = [
    #     -3.2835257689105433,
    #     -1.7120874563800257,
    #     1.6925301551818848,
    #     -1.7154505888568323,
    #     -1.6483410040484827,
    #     1.5793789625167847
    #     ]

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint", 
        "wrist_1_joint",
        "wrist_2_joint", 
        "wrist_3_joint",
        "camera_joint"
    ]

    COLL_LINKS = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "ee_link",
        "camera_mount",
        # # The following are not in the urdf, calculate them manually
        # "left_gripper",
        # "right_gripper",
    # 
    ]

    FLOOR_HEIGHT = -0.16

    def __init__(self, urdf_path: Path, root_name="ur5"):
        # self.urdf = yourdfpy.URDF.load(
        #     urdf_path, 
        #     filename_handler=partial(yourdfpy.filename_handler_magic, dir=urdf_path.parent),
        #     load_collision_meshes=True,
        #     build_collision_scene_graph=True,
        #     )
        # self.motion_planner = UR5MotionPlanning()
        self._server = None
        self.root_name = root_name
        self.collision_managers = {
            "base": tr.collision.CollisionManager(),
            "arm": tr.collision.CollisionManager(),
            "collbodies": tr.collision.CollisionManager(),
        }
        self.kinematics = UR5RobotKinematics(ee_offset='gripper_center')


    def create_robot(self, server: viser.ViserServer, root_transform: viser.SceneNodeHandle, use_visual: bool = False) -> None:
        """Put the robot in the scene, create collision managers

        Args:
            server (viser.ViserServer): scene server
            root_transform (viser.SceneNodeHandle): root transform of the robot (can be accessed through wxyz, position)
            use_visual: whether to use the collision body or the visual body for visual rendering
        """
        # root_transform is from calling add_frame
        self._server = server
        self._ur5_frame = root_transform

        left_finger, right_finger, wrist_coll, wrist_2_coll, wrist_1_coll, forearm_coll = [], [], [], [], [], []
        scene = self.urdf.scene if use_visual else self.urdf.collision_scene
        for frame_name, value in scene.geometry.items():
            assert isinstance(value, tr.Trimesh)
            orig_name = frame_name
            frame_name = scene.graph.transforms.parents[frame_name]
            full_name = self._frame_name_with_parents(frame_name)
            mesh_frame_name = full_name + "/" + orig_name
            if frame_name == "camera_mount":
                value.visual.vertex_colors = (255, 255, 255)
            # self._server.add_mesh(
            #     f"/{self.root_name}/" + mesh_frame_name,
            #     vertices=value.vertices,
            #     faces=value.faces,
            #     color=(150, 150, 150),
            # )
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            self._server.add_mesh_trimesh(
                name=f"/{self.root_name}/" + mesh_frame_name,
                mesh=value
            )
            if "robotiq" in full_name:
                # if "left_outer_knuckle" in full_name:
                if "left_inner_finger" in full_name:
                    left_finger.append(frame_name)
                # elif "right_outer_knuckle" in full_name:
                elif "right_inner_finger" in full_name:
                    right_finger.append(frame_name)
            if (frame_name == "camera_mount") or (frame_name == "wrist_3_link"):
                wrist_coll.append(frame_name)
            if (frame_name == "wrist_2_link"):
                wrist_2_coll.append(frame_name)
            if (frame_name == "wrist_1_link"):
                wrist_1_coll.append(frame_name)
            # if (frame_name == "forearm_link"):
            #     forearm_coll.append(frame_name)
        self.coll_groups = [left_finger, right_finger, wrist_coll, wrist_2_coll, wrist_1_coll] #, forearm_coll]
        
        self.goto_joints(self.UR5_HOME_JOINT)
        coll_transforms, coll_extents = self.get_coll_bbox_list() # oriented bbox transform/extents of the collision groups, in order
        coll_boxes = []
        for i, (coll_transform, coll_extent) in enumerate(zip(coll_transforms, coll_extents)):
            box = tr.creation.box(extents=coll_extent)
            coll_boxes.append(box)

        for coll_name in self.COLL_LINKS:
            children = list(filter(
                lambda name: name in self.urdf.collision_scene.geometry, 
                self.urdf.collision_scene.graph.transforms.children[coll_name]
                ))
            assert len(children) == 1
            mesh = self.urdf.collision_scene.geometry[children[0]]
            if coll_name == "camera_mount":
                mesh = tr.convex.convex_hull(mesh)
            self.collision_managers["arm"].add_object(coll_name, mesh)
            # mesh = mesh.copy()
            # mesh.vertices = tr.transformations.transform_points(mesh.vertices, self._get_world_pos_from_name(coll_name).matrix)
            # self._server.add_mesh("/coll/"+coll_name, mesh.vertices, mesh.faces)

        self.collision_managers["arm"].add_object("left_gripper", coll_boxes[0])
        self.collision_managers["arm"].add_object("right_gripper", coll_boxes[1])
        # coll_boxes[0].vertices = tr.transformations.transform_points(coll_boxes[0].vertices, coll_transforms[0])
        # coll_boxes[1].vertices = tr.transformations.transform_points(coll_boxes[1].vertices, coll_transforms[1])
        # self._server.add_mesh("/coll/"+"left_gripper", coll_boxes[0].vertices, coll_boxes[0].faces)
        # self._server.add_mesh("/coll/"+"right_gripper", coll_boxes[1].vertices, coll_boxes[1].faces)

        base_mesh = tr.creation.box(
            extents=(0.78,0.56,0.34),
            transform=[[1,0,0,0.19],[0,1,0,0],[0,0,1,-0.17-0.005],[0,0,0,1]],
        )
        table_mesh = tr.creation.box(
            bounds = [[-0.7,-0.5,-.3],[0.8,0.5,self.FLOOR_HEIGHT]]
        )
        self._server.add_mesh(
            f"/{self.root_name}/base/mesh",
            vertices=base_mesh.vertices,
            faces=base_mesh.faces,
            color=(150, 150, 150),
        )
        self._server.add_mesh(
            f"/{self.root_name}/table/mesh",
            vertices=table_mesh.vertices,
            faces=table_mesh.faces,
            color=(150, 150, 150),
            visible=False
        )
        robot_matrix = RigidTransform(
            rotation=tf.SO3(self._ur5_frame.wxyz).as_matrix(),
            translation=self._ur5_frame.position,
        ).matrix
        self.collision_managers["base"].add_object("base", base_mesh)
        self.collision_managers["base"].add_object("table", table_mesh)
        self.collision_managers["base"].set_transform("base", robot_matrix)
        self.collision_managers["base"].set_transform("table", robot_matrix)


    def _get_world_pos_from_name(self, frame_name: str) -> RigidTransform:
        """Quick [frame_name] -> [world]

        Args:
            frame_name (str): name of the frame (as defined in urdf)

        Returns:
            RigidTransform: transform from [frame_name] to [world]
        """
        transform = None
        # orig_frame_name = frame_name
        # # frame_name = self.urdf.collision_scene.graph.transforms.parents[frame_name]
        # while frame_name != self.urdf.collision_scene.graph.base_frame:
        #     parent = self.urdf.collision_scene.graph.transforms.parents[frame_name]
        #     T_parent_child = self.urdf.get_transform(frame_name, parent)
        #     T_parent_child = RigidTransform(
        #         rotation=T_parent_child[:3, :3],
        #         translation=T_parent_child[:3, 3],
        #         from_frame=frame_name,
        #         to_frame=parent,
        #     )
        #     if transform is None:
        #         transform = T_parent_child
        #     else:
        #         transform = T_parent_child * transform
        #     frame_name = parent
        base_frame = self.urdf.collision_scene.graph.base_frame
        transform = self.urdf.get_transform(frame_name, base_frame)
        transform = RigidTransform(
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            from_frame=frame_name,
            to_frame=base_frame,
        )
        transform = RigidTransform(
            rotation=tf.SO3(self._ur5_frame.wxyz).as_matrix(),
            translation=self._ur5_frame.position,
            from_frame=base_frame,
            to_frame="world",
        ) * transform

        # import pdb; pdb.set_trace()
        # transform = RigidTransform(
        #     rotation=tf.SO3(self._ur5_frame.wxyz).as_matrix(),
        #     translation=self._ur5_frame.position,
        #     from_frame="world",
        #     # to_frame=base_frame,
        #     to_frame=transform.from_frame,
        # ) * transform

        if np.isnan(transform.translation).any():
            import pdb; pdb.set_trace()
        
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
                children = list(filter(
                    lambda name: name in self.urdf.collision_scene.geometry, 
                    self.urdf.collision_scene.graph.transforms.children[item]
                    ))
                assert len(children) == 1
                bounds = self.urdf.collision_scene.geometry[children[0]].bounds
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


    def goto_joints(self, joints: np.ndarray, show_collbodies: bool = False, update_frames: bool = True):
        assert self._server is not None
        cfg={self.ARM_JOINT_NAMES[i]:joints[i] for i in range(len(joints))}
        self.urdf.update_cfg(cfg)
        if update_frames:
            for joint in self.urdf.joint_map.values():
                assert isinstance(joint, yourdfpy.Joint)
                T_parent_child = self.urdf.get_transform(joint.child, joint.parent)
                self._server.add_frame(
                    f"/{self.root_name}/" + self._frame_name_with_parents(joint.child),
                    wxyz=tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz,
                    position=T_parent_child[:3, 3],
                    show_axes=False,
                )
        if show_collbodies:
            all_transforms, all_extents = self.get_coll_bbox_list()
            for i, (transform, extent) in enumerate(zip(all_transforms, all_extents)):
                box = tr.creation.box(extent)
                box.vertices = tr.transformations.transform_points(box.vertices, transform)
                self._server.add_mesh(
                    name=f"/collision_group_{i}",
                    vertices=box.vertices,
                    faces=box.faces,
                    color=(150, 150, 150),
                )


    def check_grasp_scene_collision(self, target_joint_angles: np.ndarray, world_pointcloud: tr.PointCloud) -> bool:
        """Check if the grasp pose will result in a collision with the scene.

        Args:
            grasp_pose (RigidTransform): _description_

        Returns:
            bool: True if no collision (i.e., success), False otherwise
        """
        # start = time.time()
        self.goto_joints(target_joint_angles, update_frames=False)

        world_points = o3d.utility.Vector3dVector(world_pointcloud.vertices)

        transforms, extents = self.get_coll_bbox_list()
        for i, (trans, ext) in enumerate(zip(transforms, extents)):
            if (i == 0) or (i == 1):
                # ignore the scene-grippertip collision for now
                continue
            coll_body = o3d.geometry.OrientedBoundingBox(trans[:3, 3], trans[:3, :3], ext)
            pts = coll_body.get_point_indices_within_bounding_box(world_points)
            if len(pts) > 0:
                print("Colliding with collisionbody, ", i)
                return False
        # print(f"check_grasp_scene_collision took {time.time()-start:.3f}s")
        
        return True


    def check_robot_self_collision(self, target_joint_angles: np.ndarray) -> bool:
        """Check if the robot will collide with itself at the target joint angles.
        Also checks for base/table collisions.

        Args:
            target_joint_angles (np.ndarray): joint angles, in order specified in ARM_JOINT_NAMES

        Returns:
            bool: True if no collision (i.e., success), False otherwise
        """
        # start = time.time()
        self.goto_joints(target_joint_angles, update_frames=False)
        coll_transforms, coll_extents = self.get_coll_bbox_list() # oriented bbox transform/extents of the collision groups, in order
        for coll_name in self.COLL_LINKS:
            self.collision_managers["arm"].set_transform(coll_name, self._get_world_pos_from_name(coll_name).matrix)
        self.collision_managers["arm"].set_transform("left_gripper", coll_transforms[0])
        self.collision_managers["arm"].set_transform("right_gripper", coll_transforms[1])

        success = True

        in_collision_base, names = self.collision_managers["base"].in_collision_other(self.collision_managers["arm"], return_names=True)
        # import pdb; pdb.set_trace()
        if in_collision_base:
            print("coll in base -- ", names)
            success = False

        # Note(cmk): I removeed the FCLProc collision because the multiprocessing was causing issues...
        # but maybe it should be added back in?
        _, names = self.collision_managers["arm"].in_collision_internal(return_names=True)
        
        # import pdb; pdb.set_trace()
        for (link_1, link_2) in names:
            if "gripper" in link_1 or "gripper" in link_2:
                # gripper collisions are hard-coded with bounding boxes, can't search in urdf directly.
                # but they should have no collisions with other links
                print(link_1, link_2)
                return False
                success = False
            elif (
                (link_1 == self.urdf.collision_scene.graph.transforms.parents[link_2]) or 
                (link_2 == self.urdf.collision_scene.graph.transforms.parents[link_1])
            ):
                # ignore collisions between parent/child links
                continue
            elif (
                (link_1 == "camera_mount" and (link_2 in ["ee_link", "wrist_3_link"])) or 
                (link_2 == "camera_mount" and (link_1 in ["ee_link", "wrist_3_link"]))
            ):
                # ignore collisions between camera mount and ee_link/wrist_3_link
                continue
            else:
                # all other collisions are bad
                print(link_1, link_2)
                return False
                success = False
        # print("check_robot_self_collision took", time.time() - start, "seconds")

        return success
        
        
    def create_traj_from_grasp(self, grasp_pose: RigidTransform, world_pointcloud: tr.PointCloud) -> Tuple[Any, Any, Any]:
        """Given a grasp pose, create a trajectory to execute the grasp.
        Currently, performs the following:
         - Move to a pre-grasp position, first rotating the last three joints, then rotating the first three joints.
         The pre-grasp position is 10cm above the grasp position (in EE frame)
         - Move forwards 10cm into the grasp position.

        Args:
            grasp_pose (RigidTransform): grasp pose, in UR5 EE pose
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)

        Returns:
            Tuple[Any, Any, Any]: (trajectory, success, fin_pose) if success is True, (None, False) otherwise
                             (trajectory is np.ndarray with shape (n_steps, n_joints))
                             (fin_pose is the final pose of the EE, in UR5 EE pose)
        """
        cur_q = self.UR5_HOME_JOINT
        
        PRE_T = RigidTransform(translation=[0, 0, -0.05],from_frame=grasp_pose.from_frame, to_frame=grasp_pose.from_frame)
        pre_grasp = grasp_pose * PRE_T
        if not self._check_pose_reachable(cur_q, pre_grasp, world_pointcloud, allow_180=True):
            print("Pre-Grasp pose not reachable!")
            return None, False, None

        # At this point, we know that the pre-grasp pose is collision-free, so we can use that as a starting point.
        # Give ~2cm of leeway for the gripper to move down.
        if not self._check_pose_reachable(cur_q, grasp_pose, world_pointcloud, allow_180=True):
            pose_reachable = False
            # It might just be that for grasps close to ground, the gripper will collide with the ground if it gets to the actual grasp pose.
            # So, the grasp pose may need to be adjusted to be a bit higher.
            # for dist in np.linspace(5, 4, 5):
            # for dist in np.linspace(0, 1, 5):
            #     OFFSET_T = RigidTransform(translation=[0, 0, -0.01*dist],from_frame=grasp_pose.from_frame, to_frame=grasp_pose.from_frame)
            #     new_grasp_pose = grasp_pose * OFFSET_T
            #     if self._check_pose_reachable(cur_q, new_grasp_pose, world_pointcloud, allow_180=True):
            #         grasp_pose = new_grasp_pose
            #         pose_reachable = True
            #         break
            if not pose_reachable:
                print("Grasp pose not reachable!")
                return None, False, None

        traj, succ, has_flipped_180 = self._create_traj(cur_q, pre_grasp, world_pointcloud, allow_180=True)
        if not succ:
            return None, False, None

        if has_flipped_180:
            grasp_pose = grasp_pose * RigidTransform(
                rotation=RigidTransform.z_axis_rotation(np.pi),
                from_frame=grasp_pose.from_frame,
                to_frame=grasp_pose.from_frame
            )

        # Rotate the last two joints to the pregrasp position, then rotate the first four joints to the pregrasp position.
        traj_last_three_joints = traj.copy()
        traj_last_three_joints[:, :3] = cur_q[:3]
        traj_first_three_joints = traj.copy()
        traj_first_three_joints[:, 3:] = traj_last_three_joints[-1, 3:]

        traj_remaining, succ, _ = self._create_traj(traj_first_three_joints[-1], grasp_pose, world_pointcloud, allow_180=False)
        if not succ or True in np.isnan(grasp_pose.rotation):
            return None, False, None
        traj = np.concatenate([traj_last_three_joints, traj_first_three_joints, traj_remaining], axis=0)
        return traj, succ, grasp_pose


    def create_traj_lift_up(self, cur_q: np.ndarray, cur_pose: RigidTransform, world_up_dist: float, world_pointcloud: tr.PointCloud) -> Tuple[Any, Any]:
        """Create a trajectory to lift the gripper up by world_up_dist.

        Args:
            cur_q (np.ndarray): current joint angles
            cur_pose (RigidTransform): current pose of the gripper, in world frame
            world_up_dist (float): distance to lift up, in world frame (z-axis)
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)

        Returns:
            Tuple[Any, Any]: (trajectory, success) if success is True, (None, False) otherwise
        """
        post_grasp = RigidTransform(
            translation=[0, 0, world_up_dist],
            from_frame=cur_pose.to_frame,
            to_frame=cur_pose.to_frame
            ) * cur_pose

        traj, succ, _ = self._create_traj(cur_q, post_grasp, world_pointcloud, allow_180=False)
        if not succ:
            return None, False
        
        return traj, succ
    
    def create_traj_place(self, cur_q: np.ndarray, place_pose: RigidTransform, world_pointcloud: tr.PointCloud) -> Tuple[Any, Any]:
        """Create a trajectory to lift the gripper up by world_up_dist.

        Args:
            cur_q (np.ndarray): current joint angles
            cur_pose (RigidTransform): current pose of the gripper, in world frame
            world_up_dist (float): distance to lift up, in world frame (z-axis)
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)

        Returns:
            Tuple[Any, Any]: (trajectory, success) if success is True, (None, False) otherwise
        """
        post_grasp = place_pose

        traj, succ, _ = self._create_traj(cur_q, post_grasp, world_pointcloud, allow_180=False)
        if not succ:
            return None, False
        
        return traj, succ

    def create_traj_twist(self, cur_q: np.ndarray, cur_pose: RigidTransform, twist_dist: float, world_pointcloud: tr.PointCloud) -> Tuple[Any, Any]:
        """Create a trajectory to lift the gripper up by world_up_dist.

        Args:
            cur_q (np.ndarray): current joint angles
            cur_pose (RigidTransform): current pose of the gripper, in world frame
            world_up_dist (float): distance to lift up, in world frame (z-axis)
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)

        Returns:
            Tuple[Any, Any]: (trajectory, success) if success is True, (None, False) otherwise
        """
        # import pdb; pdb.set_trace()
        post_grasp =  cur_pose * RigidTransform(translation=[0, 0, 0],rotation=RigidTransform.z_axis_rotation(twist_dist),from_frame=cur_pose.from_frame,to_frame=cur_pose.from_frame) 

        traj, succ, _ = self._create_traj(cur_q, post_grasp, world_pointcloud, allow_180=False)
        if not succ:
            return None, False
        
        return traj, succ
    
    def create_traj_pour(self, cur_q: np.ndarray, cur_pose: RigidTransform, twist_dist: float, world_pointcloud: tr.PointCloud) -> Tuple[Any, Any]:
        """Create a trajectory to lift the gripper up by world_up_dist.

        Args:
            cur_q (np.ndarray): current joint angles
            cur_pose (RigidTransform): current pose of the gripper, in world frame
            world_up_dist (float): distance to lift up, in world frame (z-axis)
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)

        Returns:
            Tuple[Any, Any]: (trajectory, success) if success is True, (None, False) otherwise
        """
        # import pdb; pdb.set_trace()
        post_grasp =  cur_pose * RigidTransform(translation=[0, 0, 0],rotation=RigidTransform.x_axis_rotation(twist_dist),from_frame=cur_pose.from_frame,to_frame=cur_pose.from_frame) 

        traj, succ, _ = self._create_traj(cur_q, post_grasp, world_pointcloud, allow_180=False)
        if not succ:
            return None, False
        
        return traj, succ
    

    def _check_traj_collision(self, start_joint_angles: np.ndarray, end_joint_angles: np.ndarray, world_pointcloud: tr.PointCloud, num_waypoints: int = 20, only_check_end: bool = False) -> bool:
        """
        Check if a trajectory between two joint angles is collision-free, both self-collision and collision with the scene.
        if only_check_end is True, only check the end of the trajectory (i.e. end_joint_angles).
        """
        # start = time.time()
        traj = np.linspace(start_joint_angles, end_joint_angles, num_waypoints)

        # if collides with scene, return False
        scene_success = self.check_grasp_scene_collision(traj[-1], world_pointcloud)
        if not scene_success:
            print("collides with scene")
            return False

        # if collides with self, return False
        success = self.check_robot_self_collision(traj[-1])
        if only_check_end:
            print("Only checking end, currently success:", success)
            return success

        for joint_angles in traj:
            curr_success = self.check_robot_self_collision(joint_angles)
            if not curr_success:
                print("collides with itself")
                return False
            
        return True
    
        # if not success:
        #     return False

        # success = [
        #     (self.check_robot_self_collision(joint_angles)) for joint_angles in traj[:-1]
        # ]
        # Save scene-collision check for the last point.
        # success = success and 
        # # print("collision check time:", time.time() - start)
        # success = np.all(success)
        # return success

    
    def _check_pose_reachable(self, start_joint_angles: np.ndarray, end_pose: RigidTransform, world_pointcloud: tr.PointCloud, allow_180: bool = False) -> bool:
        """Check if the pose is reachable -- bypass all the trajectory generation.

        Args:
            start_joint_angles (np.ndarray): _description_
            end_pose (RigidTransform): _description_
            world_pointcloud (tr.PointCloud): _description_

        Returns:
            bool: True if reachable, False otherwise
        """
        def collide_angles(angles: np.ndarray):
            angles = angles.copy()
            for i in range(len(angles)):
                if angles[i] - start_joint_angles[i] >= np.pi:
                    angles[i] -= 2*np.pi
                elif angles[i] - start_joint_angles[i] <= -np.pi:
                    angles[i] += 2*np.pi
            safe = self._check_traj_collision(start_joint_angles, angles, world_pointcloud, only_check_end=True)
            if allow_180:
                angles_180 = angles.copy()
                angles_180[-1] += np.pi
                safe_180 = self._check_traj_collision(start_joint_angles, angles_180, world_pointcloud, only_check_end=True)
                return safe or safe_180
            return safe
            
        self.kinematics.set_config(start_joint_angles)
        try:
            self.kinematics.ik(end_pose.matrix, ee_offset='gripper_center')
        except:
            return False
        target_joint_angles = self.kinematics.config
        safe = collide_angles(target_joint_angles)

        self.kinematics.set_config(start_joint_angles)
        other_joint_angles = self.kinematics.ika8(end_pose.matrix, ee_offset='gripper_center') # ndarray (6, 8)
        safe_other = [collide_angles(other_joint_angles[:, i]) for i in range(other_joint_angles.shape[1])]
        # print(safe, safe_other)

        return safe or np.any(safe_other)
        

    def _create_traj(self, start_joint_angles: np.ndarray, end_pose: RigidTransform, world_pointcloud: tr.PointCloud, allow_180: bool = False) -> Tuple[Any, Any, Any]:
        """Create a trajectory between two joint angles.

        Args:
            start_joint_angles (np.ndarray): starting joint angles
            end_pose (RigidTransform): end EE pose (4x4 matrix)
            world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)
            allow_180 (bool, optional): whether allow rotating the gripper 180 degrees to avoid collision

        Returns:
            Tuple[Any, Any, Any]: (trajectory, success, has_flipped) if success is True, (None, False, None) otherwise
                                  (trajectory is np.ndarray with shape (n_steps, n_joints))
                                  (has_flipped is True if the gripper has flipped 180 degrees, False otherwise)
        """
        def collide_angles(angles: np.ndarray):
            angles = angles.copy()
            for i in range(len(angles)):
                if angles[i] - start_joint_angles[i] >= np.pi:
                    angles[i] -= 2*np.pi
                elif angles[i] - start_joint_angles[i] <= -np.pi:
                    angles[i] += 2*np.pi
                # if angles[i] - start_joint_angles[i] >= 2*np.pi:
                #     angles[i] -= 2*np.pi
                # elif angles[i] - start_joint_angles[i] <= -2*np.pi:
                #     angles[i] += 2*np.pi
            collides = (not self._check_traj_collision(start_joint_angles, angles, world_pointcloud))
            if allow_180:
                angles_180 = angles.copy()
                angles_180[-1] += np.pi
                collides_180 = (not self._check_traj_collision(start_joint_angles, angles_180, world_pointcloud))
                return ((collides, angles), (collides_180, angles_180))
            return (collides, angles)
            
        self.kinematics.set_config(start_joint_angles)
        self.kinematics.ik(end_pose.matrix, ee_offset='gripper_center')
        target_joint_angles = self.kinematics.config
        collides = collide_angles(target_joint_angles)

        self.kinematics.set_config(start_joint_angles)
        other_joint_angles = self.kinematics.ika8(end_pose.matrix, ee_offset='gripper_center') # ndarray (6, 8)
        collides_other = [collide_angles(other_joint_angles[:, i]) for i in range(other_joint_angles.shape[1])]

        # None of the IK solutions work.
        working_joint_angles, flipped_180 = [], []
        if allow_180:
            if collides[0][0] and collides[1][0] and all([c[0][0] and c[1][0] for c in collides_other]):
                return None, False, None
            else:
                if not collides[0][0]:
                    working_joint_angles.append(collides[0][1])
                    flipped_180.append(False)
                if not collides[1][0]:
                    working_joint_angles.append(collides[1][1])
                    flipped_180.append(True)
                for c in collides_other:
                    if not c[0][0]:
                        working_joint_angles.append(c[0][1])
                        flipped_180.append(False)
                    if not c[1][0]:
                        working_joint_angles.append(c[1][1])
                        flipped_180.append(True)
        else:
            if collides[0] and all([c[0] for c in collides_other]):
                return None, False, None
            else:
                if not collides[0]:
                    working_joint_angles.append(collides[1])
                    flipped_180.append(False)
                for c in collides_other:
                    if not c[0]:
                        working_joint_angles.append(c[1])
                        flipped_180.append(False)
        
        # Find the joint angles that are closest to the start joint angles.
        working_joint_angles = np.array(working_joint_angles)
        dists = np.linalg.norm(working_joint_angles - start_joint_angles, axis=1)
        # closest_idx_list = np.argsort(dists)[::-1]
        # for idx in closest_idx_list:
        #     # check collision here!
        #     if self._check_traj_collision(start_joint_angles, working_joint_angles[idx], world_pointcloud):
        #         closest_joint_angles = working_joint_angles[idx]
        #         flipped_180 = flipped_180[idx]
        #         break
        closest_idx = np.argmin(dists)
        closest_joint_angles = working_joint_angles[closest_idx]
        flipped_180 = flipped_180[closest_idx]
        
        traj = np.linspace(start_joint_angles, closest_joint_angles, 20)
        # import pdb; pdb.set_trace()
        return traj, True, flipped_180


    # def _create_traj(self, start_joint_angles: np.ndarray, end_pose: RigidTransform, world_pointcloud: tr.PointCloud, allow_180: bool = False) -> Tuple[Any, Any, Any]:
    #     """Create a trajectory between two joint angles.

    #     Args:
    #         start_joint_angles (np.ndarray): starting joint angles
    #         end_pose (RigidTransform): end EE pose (4x4 matrix)
    #         world_pointcloud (tr.PointCloud): pointcloud of the world (to check for collisions)
    #         allow_180 (bool, optional): whether allow rotating the gripper 180 degrees to avoid collision

    #     Returns:
    #         Tuple[Any, Any, Any]: (trajectory, success, has_flipped) if success is True, (None, False, None) otherwise
    #                               (trajectory is np.ndarray with shape (n_steps, n_joints))
    #                               (has_flipped is True if the gripper has flipped 180 degrees, False otherwise)
    #     """
    #     """
    #     allow_180: whether allow rotating the gripper 180 degrees to avoid collision 
    #     (eg. if the camera was supposed to inside, rotate it to the outside)
    #     """
    #     self.kinematics.set_config(start_joint_angles)
    #     self.kinematics.ik(end_pose.matrix, ee_offset='gripper_center')
    #     target_joint_angles = self.kinematics.config

    #     if target_joint_angles[0] - start_joint_angles[0] > np.pi:
    #         target_joint_angles[0] -= 2*np.pi
    #     elif target_joint_angles[0] - start_joint_angles[0] < -np.pi:
    #         target_joint_angles[0] += 2*np.pi

    #     collide = (not self._check_traj_collision(start_joint_angles, target_joint_angles, world_pointcloud))

    #     working_trajs = [] # store (target_joint_angles, has_flipped_180)

    #     if not collide:
    #         working_trajs.append((target_joint_angles.copy(), False))

    #     if collide and allow_180:
    #         if target_joint_angles[-1] > 0:
    #             target_joint_angles[-1] -= np.pi
    #         else:
    #             target_joint_angles[-1] += np.pi
    #         collide = (not self._check_traj_collision(start_joint_angles, target_joint_angles, world_pointcloud))
    #         if not collide:
    #             print("rotate 180 degrees to avoid collision")
    #             working_trajs.append((target_joint_angles.copy(), True))

    #     if collide:
    #         other_joint_angles = self.kinematics.ika8(end_pose.matrix, ee_offset='gripper_center') # ndarray (6, 8)
    #         for i in range(8):
    #             if other_joint_angles[0, i] - start_joint_angles[0] > np.pi:
    #                 other_joint_angles[0, i] -= 2*np.pi
    #             elif other_joint_angles[0, i] - start_joint_angles[0] < -np.pi:
    #                 other_joint_angles[0, i] += 2*np.pi

    #             collide = (not self._check_traj_collision(start_joint_angles, other_joint_angles[:, i], world_pointcloud))
    #             if not collide:
    #                 working_trajs.append((other_joint_angles[:, i].copy(), False))

    #             if collide and allow_180:
    #                 if other_joint_angles[-1, i] > 0:
    #                     other_joint_angles[-1, i] -= np.pi
    #                 else:
    #                     other_joint_angles[-1, i] += np.pi
    #                 collide = (not self._check_traj_collision(start_joint_angles, other_joint_angles[:, i], world_pointcloud))

    #                 if not collide:
    #                     working_trajs.append((other_joint_angles[:, i].copy(), True))
        
    #     if len(working_trajs) == 0:
    #         print("no working inds")
    #         return None, False, None
    #     elif len(working_trajs) == 1:
    #         print("one working ind")
    #         target_joint_angles = working_trajs[0][0]
    #         has_flipped_180 = working_trajs[0][1]
    #         traj = np.linspace(start_joint_angles, target_joint_angles, 20)
    #         return traj, True, has_flipped_180
    #     else:
    #         print("multiple working inds")
    #         # then we have multiple working inds, want to choose the one that is closest to the start_joint_angles
    #         min_dist = np.inf
    #         for curr_target_joints, has_flipped in working_trajs:
    #             dist = np.linalg.norm(start_joint_angles - curr_target_joints)
    #             if dist < min_dist:
    #                 min_dist = dist
    #                 target_joint_angles = curr_target_joints
    #                 has_flipped_180 = has_flipped
    #         traj = np.linspace(start_joint_angles, target_joint_angles, 20)
    #         return traj, True, has_flipped_180


def main(
    urdf_path: Path = Path("pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf"),
    pointcloud_path: str = "world_pointcloud.ply",
):
    """
    Simple test function for the grasp planner.
    Spins up:
     - the UR5 robot
     - controllable grasp pose (draggable as a transform controls object in viser)
     - a point cloud of the world (from the world_pointcloud.ply file)

    This function is useful for debugging the following:
     - trajectory planning
     - collision checking.

    Args:
        urdf_path (Path, optional): Path to URDF file. Defaults to Path("pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf").
        pointcloud_path (str, optional): Path to point cloud file. Defaults to "world_pointcloud.ply".
    """
    grasp_planner = UR5GraspPlanner(
        urdf_path=urdf_path,
        root_name="robot"
    )
    server = viser.ViserServer()
    server.add_frame(
        name="/world",
        axes_length=0.1,
        axes_radius=0.01,
        show_axes=True
    )
    # world_pointcloud = tr.load(pointcloud_path)
    # server.add_point_cloud(
    #     name="/world/pointcloud",
    #     points=world_pointcloud.vertices,
    #     colors=world_pointcloud.visual.vertex_colors[:, :3],
    #     point_size=0.005
    # )
    robot_frame = server.add_frame(
        name="/robot",
        wxyz=tf.SO3.from_z_radians(np.pi).wxyz,
        show_axes=False
    )
    grasp_planner.create_robot(server, robot_frame, use_visual=True)
    grasp_planner.goto_joints(grasp_planner.UR5_HOME_JOINT)
    
    grasp_pose = RigidTransform(
        rotation=np.array([
            [-0.332911, -0.423846, 0.842333],
            [-0.155789, 0.905734, 0.394177],
            [-0.930000, 0.000000, -0.367559]
        ]),
        translation=np.array([0.602751, -0.017508, -0.091149])
    )

    grasp_controls = server.add_transform_controls(
        name="/grasp",
        scale=0.1,
    )
    robot_frame_R = RigidTransform(
        rotation=RigidTransform.y_axis_rotation(np.pi/2) @ RigidTransform.z_axis_rotation(np.pi/2)
    )
    grasp_controls.wxyz = grasp_pose.quaternion
    grasp_controls.position = grasp_pose.translation
    gn_frame = server.add_frame(
        name="/grasp/gn",
        wxyz=robot_frame_R.quaternion,
        # position=np.array([0.06-0.02, 0, 0]),
        position=np.array([0.02, 0, 0]),
        axes_length=0.05,
        axes_radius=0.003,
        show_axes=False
    )
    from graspnetAPI import Grasp
    grasp_gna = Grasp()
    grasp_gna.depth = 0.02
    grasp_mesh = grasp_gna.to_open3d_geometry()
    server.add_mesh(
        name="/grasp/mesh",
        vertices=np.asarray(grasp_mesh.vertices),
        faces=np.asarray(grasp_mesh.triangles),
    )

    button = server.add_gui_button(
        label="Plan"
    )
    slider = server.add_gui_slider(
        label="traj time",
        min=0,
        max=79,
        initial_value=0,
        step=1,
    )
    traj = None
    num_rotations_test = 1

    controls_checkbox = server.add_gui_checkbox(
        label="use controls",
        initial_value=True
    )

    @button.on_click
    def _(_):
        nonlocal traj

        # robot_frame.visible = False
        if controls_checkbox.value:
            orig_grasp_pose = RigidTransform(
                translation=grasp_controls.position,
                rotation=tf.SO3(grasp_controls.wxyz).as_matrix(),
                from_frame="grasp",
                to_frame="world"
            )
        else:
            orig_grasp_pose = RigidTransform(
                translation=grasps_dict[traj_grasp_ind][0].position,
                rotation=tf.SO3(grasps_dict[traj_grasp_ind][0].wxyz).as_matrix(),
                from_frame="grasp",
                to_frame="world"
            ) 

        succ_traj_list = [] # store (traj, fin_pose)
        start = time.time()
        for i in range(num_rotations_test):
            print("Trying rotation", i)
            grasp_pose = orig_grasp_pose * RigidTransform(
                rotation=RigidTransform.y_axis_rotation(i * (2*np.pi)/num_rotations_test),
                from_frame="grasp",
                to_frame="grasp"
            ) * RigidTransform(
                translation=gn_frame.position,
                rotation=tf.SO3(gn_frame.wxyz).as_matrix(),
                from_frame="grasp/ee",
                to_frame="grasp"
            )
            if grasp_pose.matrix[:, 2][2] > 0:
                continue
            
            traj, succ, fin_pose = grasp_planner.create_traj_from_grasp(grasp_pose, world_pointcloud=world_pointcloud)
            if succ:
                succ_traj_list.append((traj, fin_pose))

        # if not succ:
        if len(succ_traj_list) == 0:
            print("None succeeded")
            traj = None
            grasp_planner.goto_joints(grasp_planner.UR5_HOME_JOINT)
            slider.value = 0
        else:
            # find the best traj!
            min_dist = np.inf
            best_traj, best_end_pose = None, None
            for curr_traj, end_pose in succ_traj_list:
                dist = np.linalg.norm(curr_traj[0, :] - curr_traj[-1, :])
                if dist < min_dist:
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
                slider.value = 0
            else:
                print("succeeded!")
                traj = np.concatenate([traj, traj_up], axis=0)
                slider.value = 79
        print("Time taken:", time.time() - start)
        # robot_frame.visible = True

    @slider.on_update
    def _(_):
        if traj is None:
            return
        grasp_planner.goto_joints(traj[slider.value])

    # grasps = GraspGroup().from_npy("many_grasps.npy")

    # grasps_dict = {}
    # traj_grasp_ind = 0
    # for ind, grasp in enumerate(grasps):
    #     def curry_grasp(grasp):
    #         default_grasp = Grasp()
    #         default_grasp.depth = grasp.depth
    #         default_grasp = default_grasp.to_open3d_geometry()
    #         # lerf_color = cmap(lerf_scores[ind])[:3]
    #         curr_ind = ind
    #         frame_handle = server.add_frame(
    #             name=f'/lerf/grasps_{ind}',
    #             wxyz=tf.SO3.from_matrix(grasp.rotation_matrix).wxyz,
    #             position=grasp.translation,
    #             show_axes=False
    #         )
    #         frame_show_handle = server.add_frame(
    #             name=f'/lerf/grasps_{ind}/axes',
    #             axes_length=0.05,
    #             axes_radius=0.002,
    #             show_axes=True,
    #             visible=False
    #         )
    #         grasp_handle = server.add_mesh(
    #             name=f'/lerf/grasps_{ind}/mesh',
    #             vertices=np.asarray(default_grasp.vertices),
    #             faces=np.asarray(default_grasp.triangles),
    #             # color=lerf_color,
    #             clickable=True
    #         )
    #         ur5_handle = server.add_frame(
    #             name=f'/lerf/grasps_{ind}/ur5',
    #             wxyz=robot_frame_R.quaternion,
    #             # position=np.array([grasp.depth-0.02, 0, 0]),
    #             position=np.array([0.03, 0, 0]),
    #             axes_length=0.05,
    #             axes_radius=0.002,
    #             show_axes=True,
    #             visible=False
    #         )
    #         @grasp_handle.on_click
    #         def _(_):
    #             nonlocal traj_grasp_ind
    #             # Would be nice to set the color of the selected grasp to something else... 
    #             print(f"Trajectory grasp set to {curr_ind}, from {traj_grasp_ind}")
    #             print(grasps[curr_ind])
    #             grasps_dict[curr_ind][1].visible = True
    #             grasps_dict[traj_grasp_ind][1].visible = False
    #             traj_grasp_ind = curr_ind
    #         return frame_handle, frame_show_handle, grasp_handle, ur5_handle
        
    #     grasps_dict[ind] = curry_grasp(grasp)

    # # set the initial traj grasp to the best grasp.
    # grasps_dict[traj_grasp_ind][1].visible = True

    # gui_joints: List[viser.GuiHandle[float]] = []
    # with server.gui_folder("Joints"):
    #     button = server.add_gui_button("Reset")

    #     @button.on_click
    #     def _(_):
    #         for g, home_value in zip(gui_joints, grasp_planner.UR5_HOME_JOINT):
    #             g.value = home_value

    #     def update_frames():
    #         target_joints = np.array([gui.value for gui in gui_joints])
    #         grasp_planner.goto_joints(target_joints)
    #         success = grasp_planner.check_robot_self_collision(target_joints)
    #         if success:
    #             print("No self-collision")
    #         else:
    #             print("Self-collision")
    #         success = grasp_planner.check_grasp_scene_collision(target_joints, world_pointcloud=world_pointcloud)
    #         if success:
    #             print("No scene collision")
    #         else:
    #             print("Scene collision")

    #     for joint_name, joint in grasp_planner.urdf.joint_map.items():
    #         assert isinstance(joint, yourdfpy.Joint)
    #         if joint_name not in grasp_planner.ARM_JOINT_NAMES:
    #             continue
    #         slider = server.add_gui_slider(
    #             name=joint_name,
    #             min=(
    #                 joint.limit.lower
    #                 if joint.limit is not None and joint.limit.lower is not None
    #                 else -np.pi
    #             ),
    #             max=(
    #                 joint.limit.upper
    #                 if joint.limit is not None and joint.limit.upper is not None
    #                 else np.pi
    #             ),
    #             step=1e-3,
    #             initial_value=0.0,
    #         )
    #         if joint.limit is None:
    #             slider.visible = False

    #         @slider.on_update
    #         def _(_):
    #             update_frames()

    #         gui_joints.append(slider)

    # update_frames()

    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)
    