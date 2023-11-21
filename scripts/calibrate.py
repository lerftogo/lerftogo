from ur5py.ur5 import UR5Robot
import cv2
from dexnerf.cameras.zed import ZedImageCapture
from dexnerf.capture.capture_utils import estimate_cam2rob
import time
import numpy as np
from autolab_core import CameraIntrinsics,PointCloud,RigidTransform,Point
import matplotlib.pyplot as plt
from dexnerf.capture.capture_utils import _generate_hemi
import subprocess
from robot_lerf.test_cam import BRIOWebcam

def find_corners(img,sx,sy,SB=True):
    '''
    sx and sy are the number of internal corners in the chessboard
    '''
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((sx * sy, 3), np.float32)
    objp[:, :2] = np.mgrid[0:sx, 0:sy].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # create images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    if SB:
        ret, corners = cv2.findChessboardCornersSB(gray, (sx, sy), None)
    else:
        ret, corners = cv2.findChessboardCorners(gray, (sx, sy), None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        if corners is not None:
            return corners.squeeze()
    return None

def register_zed():
    N = 6
    MANUAL = False
    sx = 8
    sy = 6
    zed = ZedImageCapture(exposure=-1,gain=-1)
    zed.print_camera_settings()
    img_left, img_right = zed.capture_image()
    ur = UR5Robot()
    if MANUAL:
        H_WRIST = RigidTransform(translation=[0,0,0]).as_frames('rob','rob')
    else:
        H_WRIST = RigidTransform.load('cfg/T_zed_wrist.tf').as_frames('rob','rob')
    ur.set_tcp(H_WRIST)
    H_chess_cams = []
    H_rob_worlds = []
    center = [0,.5,-.3]
    traj = _generate_hemi(.5,3,2,(np.deg2rad(140),np.deg2rad(180)),(np.deg2rad(20),np.deg2rad(50)),
        center,center)
    for p in traj:
        if MANUAL:
            ur.start_freedrive()
            input("Enter to take picture")
        else:
            ur.move_pose(p,vel=1,acc=.2)
            time.sleep(.1)
        img_left, img_right = zed.capture_image()
        img_left,img_right=np.copy(img_left),np.copy(img_right)
        l_corners = find_corners(img_left, sx, sy,True)
        r_corners = find_corners(img_right, sx, sy,True)
        print(f"Found corners: {l_corners is not None}, {r_corners is not None}")
        if l_corners is None or r_corners is None:
            _,axs=plt.subplots(1,2)
            axs[0].imshow(img_left)
            if l_corners is not None:
                axs[0].scatter(l_corners[:, 0], l_corners[:, 1], s=10)
            axs[1].imshow(img_right)
            if r_corners is not None:
                axs[1].scatter(r_corners[:, 0], r_corners[:, 1], s=10)
            plt.show()
            continue
        H_rob_world = ur.get_pose()
        camera_intr = CameraIntrinsics('zed', zed.fxl, zed.fxl, zed.cxl, zed.cyl,height=img_left.shape[0],width=img_left.shape[1])
        cam_sep = abs(zed.stereo_translation[0])  # meters
        Pl = np.zeros((3, 4))
        Pl[0:3, 0:3] = np.eye(3)
        Pl = camera_intr.proj_matrix @ Pl
        Pr = np.zeros((3, 4))
        Pr[0:3, 0:3] = np.eye(3)
        Pr[0, 3] = -cam_sep
        Pr = camera_intr.proj_matrix @ Pr
        zed_corners_3d = cv2.triangulatePoints(Pl, Pr, l_corners.T, r_corners.T)
        zed_corners_3d = zed_corners_3d[0:3, :] / zed_corners_3d[3, :]
        points_3d_plane=PointCloud(zed_corners_3d,'zed')
        X = np.c_[
            points_3d_plane.x_coords,
            points_3d_plane.y_coords,
            np.ones(points_3d_plane.num_points),
        ]
        y = points_3d_plane.z_coords
        A = X.T.dot(X)
        b = X.T.dot(y)
        w = np.linalg.inv(A).dot(b)
        n = np.array([w[0], w[1], -1])
        n = n / np.linalg.norm(n)
        mean_point_plane = points_3d_plane.mean()

        # find x-axis of the chessboard coordinates on the fitted plane
        T_camera_table = RigidTransform(
            translation=-points_3d_plane.mean().data,
            from_frame=points_3d_plane.frame,
            to_frame="table",
        )
        
        points_3d_centered = T_camera_table * points_3d_plane

        # get points along y
        coord_pos_x = int(np.floor(sx * sy / 2.0))
        coord_neg_x = int(np.ceil(sx * sy / 2.0))

        points_pos_x = points_3d_centered[coord_pos_x:]
        points_neg_x = points_3d_centered[:coord_neg_x]
        x_axis = np.mean(points_pos_x.data, axis=1) - np.mean(
            points_neg_x.data, axis=1
        )
        x_axis = x_axis - np.vdot(x_axis, n) * n
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(n, x_axis)

        # produce translation and rotation from plane center and chessboard
        # basis
        rotation_cb_camera = RigidTransform.rotation_from_axes(
            x_axis, y_axis, n
        )
        translation_cb_camera = mean_point_plane.data
        T_cb_camera = RigidTransform(
            rotation=rotation_cb_camera,
            translation=translation_cb_camera,
            from_frame="cb",
            to_frame='zed',
        )
        print(T_cb_camera)
        # UNCOMMENT THE BELOW TO SHOW THE AXES DURING CALIBRATION
        # display image with axes overlayed
        cb_center_im = camera_intr.project(
            Point(T_cb_camera.translation, frame='zed')
        )
        scale=.05
        cb_x_im = camera_intr.project(
            Point(
                T_cb_camera.translation
                + T_cb_camera.x_axis * scale,
                frame='zed',
            )
        )
        cb_y_im = camera_intr.project(
            Point(
                T_cb_camera.translation
                + T_cb_camera.y_axis * scale,
                frame='zed',
            )
        )
        cb_z_im = camera_intr.project(
            Point(
                T_cb_camera.translation
                + T_cb_camera.z_axis * scale,
                frame='zed',
            )
        )
        x_line = np.array([cb_center_im.data, cb_x_im.data])
        y_line = np.array([cb_center_im.data, cb_y_im.data])
        z_line = np.array([cb_center_im.data, cb_z_im.data])

        plt.figure(figsize=(10,10))
        plt.imshow(img_left.data)
        plt.scatter(cb_center_im.data[0], cb_center_im.data[1])
        plt.plot(x_line[:, 0], x_line[:, 1], c="r", linewidth=3)
        plt.plot(y_line[:, 0], y_line[:, 1], c="g", linewidth=3)
        plt.plot(z_line[:, 0], z_line[:, 1], c="b", linewidth=3)
        plt.axis("off")
        plt.title("Chessboard frame in camera %s" % ('zed'))
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()
        if 'y' in input("Rotate about z? y/[N]"):
            H = RigidTransform(rotation=RigidTransform.z_axis_rotation(180)).as_frames(T_cb_camera.from_frame,T_cb_camera.from_frame)
            T_cb_camera = T_cb_camera*H
        H_chess_cams.append(T_cb_camera.as_frames("cb","cam"))
        H_rob_worlds.append(H_rob_world.as_frames("rob","world"))
    H_cam_rob,H_chess_world = estimate_cam2rob(H_chess_cams,H_rob_worlds)
    #remove the pre-specified wrist transform 
    H_cam_rob = H_WRIST*H_cam_rob
    print("Estimated cam2rob:")
    print(H_cam_rob)
    print()
    print(H_chess_world)
    if 'n' not in input("Save? [y]/n"):
        H_cam_rob.save("cfg/T_zed_wrist.tf")





def rvec_tvec_to_transform(rvec, tvec):
    '''
    convert translation and rotation to pose
    '''
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame='tag', to_frame='cam')


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, tag_length, visualize=False):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''
    print(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

    if len(corners) == 0 or len(ids) == 0:
        raise Exception("No Aruco markers detected in the image!")

    # If markers are detected
    rvec, tvec = None, None
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], tag_length, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            frame = cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            if visualize:
                cv2.imshow('img', frame)
                cv2.waitKey(0)
            print("Returned")
    return frame, rvec, tvec





def register_webcam():
    MANUAL = True
    port_num = 0
    ur = UR5Robot()
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

    if MANUAL:
        H_WRIST = RigidTransform(translation=[0,0,0]).as_frames('rob','rob')
    else:
        # H_WRIST = RigidTransform(translation=[0,0,0]).as_frames('rob','rob')
        H_WRIST = RigidTransform.load('../cfg/T_webcam_wrist.tf').as_frames('rob','rob')
    ur.set_tcp(H_WRIST)
    H_chess_cams = []
    H_rob_worlds = []
    center = [0.44, 0,-.17]
    traj = _generate_hemi(.4,3,2,(np.deg2rad(-40),np.deg2rad(40)),(np.deg2rad(20),np.deg2rad(50)),
        center,center)
    for p in traj:
        if MANUAL:
            ur.start_freedrive()
            input("Enter to take picture")
        else:
            ur.move_pose(p,vel=1,acc=.2)
            time.sleep(.5)
        img = cam.get_frame()
        print("Capture success:", img is not None)
        # plt.imshow(img)
        # plt.show()
        H_rob_world = ur.get_pose()
        # BWW UR5 Webcam parameters 1920x1080 (cx, cy = 960, 540)
        # k = np.array(
        # [[1687.15501, 0., 975.122527],
        # [0., 1687.01480, 602.363238],
        # [0., 0., 1.]]
        # )
        # d = np.array([ 0.22258276, -0.72646325, -0.00196907,  0.00232926,  1.10562266])
        intr = CameraIntrinsics('blah',1426.4286166188251,1424.975797115347,cx=798.0496334804038,cy=469.6107388826004)
        k = intr.K
        # k = np.array(
        # [[1129.551243094171, 0., 966.9812584534886],
        # [0., 1124.5757372398643, 556.5882496966005],
        # [0., 0., 1.]]
        # )
        d = np.array([ 0., 0, 0, 0, 0])
        # tag dimensions
        l = 0.1558

        output, rvec, tvec = pose_estimation(img, cv2.aruco.DICT_ARUCO_ORIGINAL, k, d, l, False)
        T_cb_camera = rvec_tvec_to_transform(rvec, tvec)
        print("T_cb_camera", T_cb_camera)


        H_chess_cams.append(T_cb_camera.as_frames("cb","cam"))
        H_rob_worlds.append(H_rob_world.as_frames("rob","world"))
    H_cam_rob,H_chess_world = estimate_cam2rob(H_chess_cams,H_rob_worlds)
    #remove the pre-specified wrist transform 
    H_cam_rob = H_WRIST*H_cam_rob
    print("Estimated cam2rob:")
    print(H_cam_rob)
    print()
    print(H_chess_world)
    if 'n' not in input("Save? [y]/n"):
        H_cam_rob.save("T_webcam_wrist.tf")

if __name__=='__main__':
    register_webcam()