from ur5py.ur5 import UR5Robot
import cv2
from autolab_core import CameraIntrinsics,RigidTransform
import time
import torch
import multiprocessing as mp
from robot_lerf.test_cam import BRIOWebcam
from kornia.filters import laplacian
from robot_lerf.capture_utils import get_blends,ZoeProcess,_generate_hemi,_generate_traj,save_data,HOME_POS
from robot_lerf.dinoProcess import DinoProcess
# from robot_lerf.clip_process import ClipProcess
import tyro
H_CAM_WRIST = RigidTransform.load('cfg/T_webcam_wrist.tf')


def main(save_dir:str, rob: UR5Robot = None):
    mp.set_start_method('spawn')
    depth_q = mp.Queue()
    dino_q = mp.Queue()
    # clip_q_odd = mp.Queue()
    # clip_q_even = mp.Queue()
    # clip_p_even = ClipProcess(clip_q_even,'cuda:2')
    # clip_p_odd = ClipProcess(clip_q_odd,'cuda:1')
    dino_p = DinoProcess(dino_q)
    zoe_p = ZoeProcess(depth_q)
    zoe_p.start()
    dino_p.start()
    # clip_p_even.start()
    # clip_p_odd.start()

    if rob is None:
        rob = UR5Robot()

    rob.set_tcp(H_CAM_WRIST)
    time.sleep(1)

    rob.move_joint(HOME_POS,vel=.4,asyn=False)
    
    cam = BRIOWebcam('/dev/video0',1600,896)
    cam.set_value('auto_exposure',1)#1 for manual, 3 for auto
    cam.set_value('exposure_time_absolute',24)
    cam.set_value('gain',160)
    cam.set_value('exposure_dynamic_framerate',0)
    cam.set_value('backlight_compensation',0)
    cam.set_value('focus_automatic_continuous',0)
    cam.set_value('focus_absolute',14)
    cam.set_value('white_balance_automatic',0)
    cam.set_value('white_balance_temperature',4000)
    #intr for 896
    intr = CameraIntrinsics('blah',1426.4286166188251,1424.975797115347,cx=798.0496334804038,cy=469.6107388826004)
    k1=0.1638178621703512
    k2=-0.30521813423181116
    p1=-0.0011098224228067706
    p2=-0.0016512512572957351
    
    look_pos =  [0.45, 0, -0.15]#usually z set to -.15
    # center = [0.32, 0,-0.15]
    # traj = _generate_hemi(.53,16,3,(np.deg2rad(-100),np.deg2rad(100)),(np.deg2rad(70),np.deg2rad(35)),look_pos,center)
    path_1 = [(0.230, -0.58, 0.1), (0.7, -0.3, 0.1), (0.82, 0.0, 0.1), (0.7, 0.3, 0.1), (0.230, 0.58, 0.1)]
    path_2 = [(0.26, 0.4, 0.3), (0.66, 0.35, 0.3), (0.8, 0.0, 0.3), (0.66, -0.35, 0.3), (0.26, -0.4, 0.3)]
    path_3 = [(0.3, -0.27, 0.4), (0.6, -0.2, 0.4), (0.68, 0.0, 0.4), (0.6, 0.2, 0.4), (0.3, 0.27, 0.4)]

    # for a partial scan (half)
    # path_1 = [(0.230, -0.58, 0.1), (0.7, -0.3, 0.1), (0.82, 0.0, 0.1)]
    # path_2 = [(0.8, 0.0, 0.3), (0.66, -0.35, 0.3), (0.26, -0.4, 0.3)]
    # path_3 = [(0.3, -0.27, 0.4), (0.6, -0.2, 0.4), (0.68, 0.0, 0.4),]

    # # for a partial scan (3/4)
    # path_1 = [(0.230, -0.58, 0.1), (0.7, -0.3, 0.1), (0.82, 0.0, 0.1), (0.7, 0.3, 0.1)]
    # path_2 = [(0.66, 0.35, 0.3), (0.8, 0.0, 0.3), (0.66, -0.35, 0.3), (0.26, -0.4, 0.3)]
    # path_3 = [(0.3, -0.27, 0.4), (0.6, -0.2, 0.4), (0.68, 0.0, 0.4), (0.6, 0.2, 0.4)]

    final_path = path_1+path_2+path_3
    # raised_final_path = [(p[0],p[1],p[2]+.1) for p in final_path]
    traj = _generate_traj(final_path, look_pos)
    while not rob.is_stopped():
        time.sleep(.05)
    
    rob.move_pose(traj[0],vel=.5)
    input("enter to move to traj")
    if not rob.ur_c.isConnected():
        print("reconnecting")
        rob.ur_c.reconnect()
    rob.move_tcp_path(traj,vels=[.15]*len(traj),accs=[1.1]*len(traj),blends = get_blends(traj),asyn=True)
    time.sleep(1)
    all_imgs,all_poses=[],[]
    i = 0


    while not rob.is_stopped():
        pose_before = rob.get_pose()
        print(f"Image {i}, time {time.time()}")
        cam.frame=None
        while True:
            img = cam.get_frame()
            if img is not None:
                break
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # if i%2 == 0:
        #     clip_p_even.in_queue.put(img)
        # else:
        #     clip_p_odd.in_queue.put(img)
        zoe_p.in_queue.put(img)
        dino_p.in_queue.put(img)
        all_imgs.append(img)
        all_poses.append(pose_before)

        i += 1
        time.sleep(.25)

    zoe_p.kill()
    dino_p.kill()
    # clip_p_even.kill()
    # clip_p_odd.kill()
    # zoe_p.in_queue.close()
    # dino_p.in_queue.close()
    # clip_p_even.in_queue.close()
    # clip_p_odd.in_queue.close()
    # zoe_p.in_queue.join_thread()
    # dino_p.in_queue.join_thread()
    # clip_p_even.in_queue.join_thread()
    # clip_p_odd.in_queue.join_thread()

    dino_batch, depth_batch = [],[]
    for i in range(len(all_imgs)):
        depth_batch.append(depth_q.get())
    
    for i in range(len(all_imgs)):
        dino_batch.append(torch.from_numpy(dino_q.get()))

    # clip_batch = []
    # for i in range(len(all_imgs)):
    #     if i%2 == 0:
    #         img_i_scales = clip_q_even.get()
    #     else:
    #         img_i_scales = clip_q_odd.get()
    #     img_i_scales = [s for s in img_i_scales]
    #     clip_batch.append(img_i_scales)
    clear_imgs,clear_depth,clear_poses,clear_dino,clear_clip = [],[],[],[],[]
    img_tens = [torch.from_numpy(img) for img in all_imgs]
    blurs = laplacian(torch.stack(img_tens,dim=0).cuda().permute(0,3,1,2).float(),5).view(len(all_imgs),-1).var(-1)
    for i in range(len(all_imgs)):
        if blurs[i]>1.0:
            clear_imgs.append(all_imgs[i])
            clear_depth.append(depth_batch[i])
            clear_poses.append(all_poses[i])
            clear_dino.append(dino_batch[i])
            # clear_clip.append(clip_batch[i])
    print("saving ",len(clear_imgs),"clear images")
    #clear_clip is a 2D array of format [ [ [scale1],[scale2],... ], [ [scale1],[scale2],... ], ...
    save_data(clear_imgs,clear_poses,save_dir,intr,k1,k2,p1,p2,depth=clear_depth,dino=clear_dino)
            # ,clip=clear_clip, clip_level_json = [clip_p_even.get_level_json(i) for i in range(len(clear_clip[0]))], clip_pyramid_json=clip_p_even.get_pyramid_json(),clipstring = clip_p_even.clip_model_name())


if __name__=="__main__":
    tyro.cli(main)