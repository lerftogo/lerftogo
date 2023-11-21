import numpy as np
from autolab_core import CameraIntrinsics,RigidTransform
import time
import torch
import multiprocessing as mp
import queue
from scipy.spatial.transform import Rotation as R
from typing import List
from PIL import Image
import json

HOME_POS = [-4.295879427586691, -0.8578832785235804, 1.434835433959961, -2.0161431471454065, -1.639289681111471, 2.0967414379119873]

class ZoeProcess(mp.Process):
    def __init__(self,out_queue):
        super().__init__()
        self.out_queue = out_queue
        self.in_queue = mp.Queue(maxsize=0)
        self.device = "cuda:0"
        self.daemon=True
        

    def run(self):
        repo = "isl-org/ZoeDepth"
        self.zoe = torch.compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(self.device))
        self.zoe.infer(torch.ones((1,3,256,256),device='cuda'))
        running=True
        while running:
            img_batch = []
            while True:
                try:
                    img = self.in_queue.get(timeout=.01)
                except queue.Empty:
                    break
                if img is None:
                    running = False
                    print("ZOE DONE")
                    break
                img = torch.from_numpy(img).to(self.device).float() / 255.0
                img = torch.permute(img, (2, 0, 1))
                img_batch.append(img)
            if len(img_batch) == 0:
                continue
            with torch.no_grad():
                start=time.time()
                img_batch = torch.stack(img_batch)
                print("Zoeprocess batch size", img_batch.shape)
                res = self.zoe.infer(img_batch).cpu().numpy()
                for i in range(res.shape[0]):
                    self.out_queue.put(res[i,...].transpose(1,2,0))
                print(f"Zoe took {time.time()-start} seconds")

    def kill(self):
        self.in_queue.put(None)

def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    '''
    cam_t: numpy array of 3D position of gripper
    obstacle_t: numpy array of 3D position of location to point camera at
    '''
    dir=obstacle_t-cam_t
    z_axis=dir/np.linalg.norm(dir)
    #change the line below to be negative if the camera is difficult to position
    x_axis_dir = -np.cross(np.array((0,0,1)),z_axis)
    if np.linalg.norm(x_axis_dir)<1e-10:
        x_axis_dir=np.array((0,1,0))
    x_axis=x_axis_dir/np.linalg.norm(x_axis_dir)
    y_axis_dir=np.cross(z_axis,x_axis)
    y_axis=y_axis_dir/np.linalg.norm(y_axis_dir)
    #postmultiply the extra rotation to rotate the camera WRT itself
    R = RigidTransform.rotation_from_axes(x_axis,y_axis,z_axis)@extra_R
    H = RigidTransform(translation=cam_t,rotation=R,from_frame='camera',to_frame='base_link')
    #rotate by extra_R which can specify a rotation for the camera
    return H

def save_data(imgs,poses,savedir,intr:CameraIntrinsics,k1,k2,p1,p2,depth=[], dino=[],clip=[],clip_level_json=[],clip_pyramid_json=None,clipstring=None):
    '''
    takes in a list of numpy arrays and poses and saves a nerf dataset in savedir
    '''
    assert len(depth)==0 or len(depth)==len(imgs),"either provide no depth or 1 depth image per rgb image"
    assert len(dino)==0 or len(dino)==len(imgs),"either provide no dino or 1 dino image per rgb image"
    import os
    os.makedirs(savedir,exist_ok=True)
    dataset_name = savedir.split("/")[-1]
    os.makedirs(f'outputs/{dataset_name}',exist_ok=True)
    data_dict = dict()
    data_dict['frames']=[]
    data_dict['fl_x']=intr.fx
    data_dict['fl_y']=intr.fy
    data_dict['cx']=intr.cx
    data_dict['cy']=intr.cy
    data_dict['h']=imgs[0].shape[0]
    data_dict['w']=imgs[0].shape[1]
    data_dict['k1']=k1
    data_dict['k2']=k2
    data_dict['p1']=p1
    data_dict['p2']=p2
    pil_images=[]
    dino_name = "dino_vitb8"
    for i,(im,p) in enumerate(zip(imgs,poses)):
        #if RGBA, strip the alpha channel out
        if im.shape[2]==4:im=im[...,:3]
        img=Image.fromarray(im)
        pil_images.append(img)
        img.save(f'{savedir}/img{i}.jpg')
        mat=(p*RigidTransform(RigidTransform.x_axis_rotation(np.pi),to_frame=p.from_frame,from_frame=p.from_frame)).matrix
        frame = {'file_path':f'img{i}.jpg', 'transform_matrix':mat.tolist()}
        if len(depth)>0:
            frame["depth_file_path"]=f"depth{i}.npy"
            np.save(f"{savedir}/depth{i}.npy",depth[i])
        data_dict['frames'].append(frame)
    if len(dino)>0:
        dino = torch.stack(dino)
        np.save(f"outputs/{dataset_name}/dino_{dino_name}.npy", dino.cpu())
        with open(f"outputs/{dataset_name}/dino_{dino_name}.info",'w') as f:
            f.write('{"image_shape": [' +f'{data_dict["h"]},'+f' {data_dict["w"]}]'+'}')
    if len(clip)>0:
        os.makedirs(f'outputs/{dataset_name}/clip_{clipstring}',exist_ok=True)
        for scale_i in range(len(clip[0])):
            scale_imgs = [img[scale_i] for img in clip]
            scale_imgs = np.stack(scale_imgs)
            np.save(f"outputs/{dataset_name}/clip_{clipstring}/level_{scale_i}.npy", scale_imgs)
            with open(f"outputs/{dataset_name}/clip_{clipstring}/level_{scale_i}.info",'w') as f:
                f.write(clip_level_json[scale_i])
        with open(f"outputs/{dataset_name}/clip_{clipstring}.info",'w') as f:
            f.write(clip_pyramid_json)
    with open(f"{savedir}/transforms.json",'w') as fp:
        json.dump(data_dict,fp)

def _generate_traj(waypoints, look_pos, extra_R = np.eye(3)):
    '''
    waypoints: list of 3D positions in world coordinates
    '''
    l_tcp_frame = 'l_tcp'
    base_frame = 'base_link'
    traj=[]
    ps=[]
    for i in range(len(waypoints)-1):
        last = False
        if i == len(waypoints)-2:
            last = True
        start=waypoints[i]
        end=waypoints[i+1]
        #Linspace, or interpolation
        traj.append(np.linspace(start,end,5, endpoint=last))
    traj=np.concatenate(traj)
    for pose in traj:
        ps.append(point_at(pose,look_pos,extra_R = extra_R).as_frames(l_tcp_frame,base_frame))

    return ps

def _generate_hemi(R, theta_N, phi_N, th_bounds, phi_bounds, look_pos, center_pos, th_first=True, extra_R = np.eye(3)):
    '''
    R: radius of sphere
    theta_N: number of points around the z axis
    phi_N: number of points around the elevation axis
    look_pos: 3D position in world coordinates to point the camera at
    center_pos: 3D position in world coords to center the hemisphere
    '''
    l_tcp_frame = 'l_tcp'
    base_frame = 'base_link'
    poses=[]
    if th_first:
        for phi_i,phi in enumerate(np.linspace(*phi_bounds,phi_N)):
            ps=[]
            for th_i,th in enumerate(np.linspace(*th_bounds,theta_N)):
                point_x = center_pos[0] + R*np.cos(th)*np.sin(phi)
                point_y = center_pos[1] + R*np.sin(th)*np.sin(phi)
                point_z = center_pos[2] + R*np.cos(phi)
                point =np.array((point_x,point_y,point_z))
                ps.append(point_at(point,look_pos,extra_R = extra_R).as_frames(l_tcp_frame,base_frame))
            #every odd theta, reverse the direction so that the resulting traj is relatively smooth
            if phi_i%2==1:
                ps.reverse()
            poses.extend(ps)
        return poses
    else:
        for th_i,th in enumerate(np.linspace(*th_bounds,theta_N)):
            ps=[]
            for phi in np.linspace(*phi_bounds,phi_N):
                point_x = center_pos[0] + R*np.cos(th)*np.sin(phi)
                point_y = center_pos[1] + R*np.sin(th)*np.sin(phi)
                point_z = center_pos[2] + R*np.cos(phi)
                point =np.array((point_x,point_y,point_z))
                ps.append(point_at(point,look_pos,extra_R = extra_R).as_frames(l_tcp_frame,base_frame))
            #every odd theta, reverse the direction so that the resulting traj is relatively smooth
            if th_i%2==1:
                ps.reverse()
            poses.extend(ps)
        return poses

def get_blends(traj:List[RigidTransform]):
    blends=[]
    for i in range(1,len(traj)-1):
        #blend is set to min distance between traj[i].translation and its neighbors
        blends.append(min(np.linalg.norm(traj[i].translation-traj[i-1].translation),np.linalg.norm(traj[i].translation-traj[i+1].translation)))
    blends = [0]+blends+[0]
    return blends