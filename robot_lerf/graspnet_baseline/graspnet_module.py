import os
import sys

import trimesh as tr
import open3d as o3d

from typing import List
import time

import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnetAPI import GraspGroup, Grasp
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

class GraspNetModule:
    num_point_in_pc: int = 100000

    def init_net(self, ckpt_path, global_pointcloud, cylinder_radius=0.03, floor_height=-0.17):
        # Init the model
        net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=12,#was 4
            cylinder_radius=cylinder_radius,
            hmin=-0.02,
            # hmax_list=[0.03, 0.04], # hmax_list= [0.01,0.02,0.03,0.04], # 
            hmax_list=[0.01,0.02,0.03,0.04],
            is_training=False
            )
        net.to(device)
        ckpt = torch.load(ckpt_path)
        net.load_state_dict(ckpt['model_state_dict'])
        net.eval()
        self._graspnet = net

        self.floor_height = floor_height
        self.pointcloud_vertices = global_pointcloud.vertices.copy()
        self.pointcloud_vertices = self.pointcloud_vertices[self.pointcloud_vertices[:, 2] > self.floor_height+0.01]
        self.mfcdetector = ModelFreeCollisionDetector(self.pointcloud_vertices, voxel_size=0.005)

    def __call__(self, pointcloud_list: List[tr.Trimesh]) -> List[GraspGroup]:
        points_list = []
        for pointcloud in pointcloud_list:
            points = pointcloud.vertices

            if len(points) >= self.num_point_in_pc:
                idxs = np.random.choice(len(points), self.num_point_in_pc, replace=False)
            else:
                idxs1 = np.arange(len(points))
                idxs2 = np.random.choice(len(points), self.num_point_in_pc-len(points), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            points = points[idxs]

        # convert data
            end_points = dict()
            points = torch.from_numpy(points[np.newaxis].astype(np.float32)).to(device)
            points_list.append(points)

        end_points['point_clouds'] = torch.cat(points_list, dim=0)
        gg_list = self._get_grasps(end_points)
        return gg_list

    def _get_grasps(self, end_points) -> List[GraspGroup]:
        # Forward pass
        batch_size = end_points['point_clouds'].shape[0]
        grasp_list = [] # going to be List[GraspGroup]
        with torch.no_grad():
            end_points = self._graspnet(end_points)
            grasp_preds = pred_decode(end_points)
        for i in range(batch_size):
            grasp_list.append(GraspGroup(grasp_preds[i].detach().cpu().numpy()))
        return grasp_list

    @staticmethod
    def get_bbox_from_grasp(gg: Grasp):
        center = gg.translation
        R = gg.rotation_matrix
        H= np.eye(4)
        H[:3,:3] = R
        H[:3,3] = center
        extent=np.array((gg.depth,gg.width,gg.height))
        box = o3d.geometry.OrientedBoundingBox(center,H[:3,:3],extent)
        return box

    def local_collision_detection(self, gg):
        start = time.time()
        meshes = gg.to_open3d_geometry_list()
        collision_with_ground_mask = np.array([np.asarray(mesh.vertices)[:, 2].min() < self.floor_height for mesh in meshes])
        # too_high_grasp = np.array([np.asarray(mesh.vertices)[:, 2].max() > self.floor_height+0.2 for mesh in meshes])
        collision_mask = collision_with_ground_mask # | too_high_grasp
        print('starting with', len(gg))
        print('collision with ground time: ', time.time()-start, 'remaining', len(gg) - sum(collision_mask))

        start = time.time()
        collides, empty_mask = self.mfcdetector.detect(gg, collision_thresh=0.005, return_empty_grasp=True)
        collision_mask = collision_mask | (collides | empty_mask)
        print('collision detection time: ', time.time()-start, 'remaining', len(gg) - sum(collision_mask), sum(collides), sum(empty_mask))
        # collision_with_ground_mask = (gg.translations + gg.rotation_matrices[:, 0]*gg.depths[:, None])[:, 2] < self.floor_height

        start = time.time()
        no_includes_pc = []
        verts_o3d = o3d.utility.Vector3dVector(self.pointcloud_vertices)
        for grasp in gg:
            bbox = self.get_bbox_from_grasp(grasp)
            no_includes_coll_pointcloud = (len(bbox.get_point_indices_within_bounding_box(verts_o3d)) <= 10)
            no_includes_pc.append(no_includes_coll_pointcloud)
        no_includes_pc = np.stack(no_includes_pc, axis=0)
        collision_mask = collision_mask | no_includes_pc
        print('collision with pc time: ', time.time()-start, 'remaining', len(gg) - sum(collision_mask))

        gg = gg[~collision_mask]
        return gg