import multiprocessing as mp
import queue

import numpy as np
import trimesh
from urdfpy import URDF
from visualization import Visualizer3D as vis3d
import time

class CollisionChecker:

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint", 
        "wrist_1_joint",
        "wrist_2_joint", 
        "wrist_3_joint",
        "camera_joint"
    ]

    ARM_LINK_NAMES = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "ee_link",
        "left_gripper",
        "right_gripper",
        "camera_mount"
    ]

    SAFE_COLLISION_LINKS = [
        ('ee_link', 'wrist_3_link'),
        ('camera_mount', 'ee_link'), 
        ('forearm_link', 'upper_arm_link'), 
        ('camera_mount', 'wrist_3_link'), 
        ('wrist_2_link', 'wrist_3_link'), 
        ('forearm_link', 'wrist_1_link'),
        ('camera_mount', 'left_gripper'),
        ('camera_mount', 'right_gripper'),
        # ("table","left_gripper"),
        # ("table","right_gripper"),
        ("base","base_link")
    ]

    def __init__(self, n_proc):
        # self.robot = URDF.load("/home/lawrence/dmodo/ur5_go_old/ur5_pybullet/urdf/real_arm_w_camera.urdf")
        # self.robot = URDF.load('/home/lawrence/robotlerf/ur5bc/ur5/real_arm_no_offset.urdf')
        self.robot = URDF.load('/home/lawrence/robotlerf/robot_lerf/pybullet_ur5_robotiq/urdf/ur5_robotiq_85.urdf')
        self._n_proc = n_proc
        self.base_mesh = trimesh.creation.box(
            extents=(0.78,0.56,0.34),
            transform=[[1,0,0,-0.19],[0,1,0,0],[0,0,1,-0.173],[0,0,0,1]],
        )
        self.table_mesh = trimesh.creation.box(
            bounds = [[-0.5,-0.5,-.3],[0.5,0.5,-0.18]]
            # extents=(1,1,.1),
            # transform=[[1,0,0,.45],[0,1,0,0],[0,0,1,-0.18-.05],[0,0,0,1]],
        )
        self.output_queue = mp.Queue()
        self.coll_procs = []
        for i in range(self._n_proc):
            self.coll_procs.append(
                FCLProc(
                    self.robot.links,
                    self.ARM_LINK_NAMES,
                    self.base_mesh,
                    self.table_mesh,
                    self.output_queue,
                )
            )
            self.coll_procs[-1].daemon = True
            self.coll_procs[-1].start()

    def in_collision(self, q):
        num_q, num_j = q.shape
        cfg = {self.ARM_JOINT_NAMES[i]: q[:, i] for i in range(num_j)}

        link_poses = self.robot.link_fk_batch(cfgs=cfg, use_names=True)

        for i in range(self._n_proc):
            self.coll_procs[i].collides(
                link_poses,
                np.arange(
                    i * num_q // self._n_proc,
                    (i + 1) * num_q // self._n_proc,
                ),
                safe_collisions=self.SAFE_COLLISION_LINKS,
            )

        # collect computed iks
        collides = False
        for _ in range(self._n_proc):
            collides |= self.output_queue.get(True)[0]
            if collides:
                break

        return collides

    def vis_init(self):
        vis3d.figure()
        vis3d.mesh(self.base_mesh)
        vis3d.show(asynch=True,animate=False)


        self.node_list = {}
        for link in self.robot.links:
            if link.name in self.ARM_LINK_NAMES:
                # print("pose",link,pose,pose.shape)
                n = vis3d.mesh(link.collision_mesh)
                self.node_list[link.name]=n


    def vis(self,q):
        num_q, num_j = q.shape
        cfg = {self.ARM_JOINT_NAMES[i]: q[:, i] for i in range(num_j)}
        link_poses = self.robot.link_fk_batch(cfgs=cfg, use_names=True)

        # print("link_poses",link_poses["base_link"].shape)
        
        # self.robot.animate(cfg_trajectory=cfg)
        # print("qshape",num_q,num_j)

        # vis3d.figure()
        # vis3d.mesh(self.base_mesh)
        # vis3d.show(asynch=True,animate=False)


        # node_list = {}
        # for link, pose in link_poses.items():
        #     if link in self.ARM_LINK_NAMES:
        #         # print("pose",link,pose,pose.shape)
        #         n = vis3d.mesh(self.robot.link_map[link].collision_mesh,T_mesh_world= pose[0])
        #         node_list[link]=n
        # import pdb;pdb.set_trace()
        fps = 125.0
        for i in range(num_q):
            time.sleep(1.0 / fps)
            for link, pose in link_poses.items():
                if link in self.ARM_LINK_NAMES:
                   self.node_list[link]._matrix=pose[i]

        


class FCLProc(mp.Process):
    """
    Used for finding collisions in parallel using FCL.
    """

    def __init__(self, links, arm_links, base_mesh, table_mesh, output_queue):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.links = links
        self.arm_links = arm_links
        self.base_mesh = base_mesh
        self.table_mesh = table_mesh
        self.output_queue = output_queue
        self.input_queue = mp.Queue()

    def _collides(self, link_poses, inds, safe_collisions):
        """ computes collisions."""
        collides = False
        for i in inds:
            for link, pose in link_poses.items():
                if link in self.arm_links:
                    self.arm_mgr.set_transform(link, pose[i])
            _, names = self.base_mgr.in_collision_other(
                self.arm_mgr, return_names=True
            )
            for name in names:
                if name not in safe_collisions:
                    print(name)
                    collides = True
                    break
            
            is_collision, names = self.arm_mgr.in_collision_internal(return_names=True)
            for name in names:
                if name not in safe_collisions:
                    print(name)
                    collides = True
                    break


        return collides

    def run(self):
        """
        the main function of each FCL process.
        """
        self.arm_mgr = trimesh.collision.CollisionManager()
        self.base_mgr = trimesh.collision.CollisionManager()

        self.base_mgr.add_object("base",self.base_mesh)
        self.base_mgr.add_object("table",self.table_mesh)
        for link in self.links:
            if link.name in self.arm_links:
                self.arm_mgr.add_object(link.name, link.collision_mesh)

        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue
            if request[0] == "collides":
                self.output_queue.put((self._collides(*request[1:]),))

    def collides(self, link_poses, inds, pind=None, safe_collisions=[]):
        self.input_queue.put(("collides", link_poses, inds, safe_collisions))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()

    # conf = np.array([[-190/180*np.pi, -102/180*np.pi, 141/180*np.pi, -130/180*np.pi, -np.pi/2, 168/180*np.pi]])
    conf = np.array([[-179/180*np.pi, -90/180*np.pi, 127/180*np.pi, -125/180*np.pi, -160/180*np.pi, 184/180*np.pi]])
    # conf = np.array([[-179/180*np.pi, -65/180*np.pi, 110/180*np.pi, 75/180*np.pi, -125/180*np.pi, 195/180*np.pi]])


    ur_collision = CollisionChecker(n_proc=1)
    print("Hit=", ur_collision.in_collision(conf))
    if ur_collision.in_collision(conf):
        print("collision detected!")
    if args.vis:
        ur_collision.vis_init()
        ur_collision.vis(conf)

    for p in ur_collision.coll_procs:
        p.terminate()
        p.join(timeout=1.0)

