import torch
import lietorch
from lietorch import SE3
import numpy as np
import cv2
import open3d as o3d

from PIL import Image
import torchvision.transforms as T

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process

from Models.mapping_utils import *
from Models.model_utils import *
from Models.ConvBKI import *
from visualization import create_point_actor
import droid_backends

class Droid:
    def __init__(self, args, model_params, NUM_CLASSES, ignore_labels, device="cuda"):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        self.i = 0

        # visualizer
        if not self.disable_vis:
        # if True:
            print("Visualizer!")
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        self.grid_params = model_params["test"]["grid_params"]
        LOAD_EPOCH = model_params["load_epoch"]
        LOAD_DIR = model_params["save_dir"]

        # create semantic mapper 
        self.map_object = GlobalMap(
            torch.tensor([int(p) for p in self.grid_params['grid_size']], dtype=torch.long).to(device),  # Grid size
            torch.tensor(self.grid_params['min_bound']).to(device),  # Lower bound
            torch.tensor(self.grid_params['max_bound']).to(device),  # Upper bound
            torch.load(os.path.join("/opt/Volume-DROID/DROID-SLAM/droid_slam/Models", "Weights", LOAD_DIR, "filters" + str(LOAD_EPOCH) + ".pt")), # Filters
            model_params["filter_size"], # Filter size
            num_classes=NUM_CLASSES,
            ignore_labels = ignore_labels, # Classes
            device=device # Device
        )

        # track the last timestamp 
        self.last_timestamp = 0

        self.point_cloud_gen = o3d.geometry.PointCloud()


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def generate_pinhole_pointcloud(self, rgb_frame, depth_frame, se3_pose):
        """Uses purely azimuth, elevation, and depth data from camera horizontal and vertical FOV.
        Combines with odometry/pose SE3 data to give XYZRGB pointcloud in world frame. Does not convert points to camera frame
        using camera model calculation.

        rgbd_mat: image file, hxwx3, depth: hxwx1; hxwx4
        se3_pose: SE3 matrix 4x4
        return the global point cloud"""

        transform = T.ToPILImage()
        intrinsics_vec=[320.0, 320.0, 320.0, 240.0]
        fx, fy, cx, cy = intrinsics_vec[0], intrinsics_vec[1], intrinsics_vec[2], intrinsics_vec[3]

        Kh = np.array([[fx, 0, cx, 0],
              [0, fy, cy, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
        Kh_inv = np.array([[1/fx, 0, -cx/fx, 0],
                           [0, 1/fy, -cy/fy, 0],
                           [0, 0,   1,      0],
                           [0, 0,   0,      1]])
        S_inv = np.linalg.inv(se3_pose)


        # print("rgb shape before resize: " + str(rgb_frame.shape))
        # print("depth shape before resize: " + str(depth_frame.shape))

        # resize rgb to be the same shape as depth image 
        rgb_frame = transform(rgb_frame)
        rgb_frame = rgb_frame.resize((depth_frame.shape[1], depth_frame.shape[0]), Image.Resampling.LANCZOS)
        rgb_frame = np.array(rgb_frame)

        # print("rgb shape after resize: " + str(rgb_frame.shape))
        # print("depth shape after resize: " + str(depth_frame.shape))

        # Get the height and width of the image
        height, width, _ = rgb_frame.shape

        # Create arrays of pixel coordinates
        x, y = np.arange(width), np.arange(height)
        
        # Reshape the pixel coordinates and depth into vectors
        x = x.reshape(-1)
        y = y.reshape(-1)

        depth_z = depth_frame.cpu().numpy()
        depth_z = depth_z.reshape(-1)

        x_proj, y_proj = np.meshgrid(x, y)
        x_proj = x_proj.reshape(-1)
        y_proj = y_proj.reshape(-1)

        onesarr = np.ones(x_proj.shape)

        # print("x_proj shape: " + str(x_proj.shape))
        # print("y_proj shape: " + str(y_proj.shape))
        # print("depth_z shape: " + str(depth_z.shape))
        # print("onesarr shape: " + str(onesarr.shape))

        # pvec = np.vstack((x, y, np.ones(x_proj.shape).transpose(), 1/depth_z)).transpose()
        np.seterr(divide='ignore', invalid='ignore')
        pvec = np.vstack((x_proj, y_proj, onesarr, 1/depth_z))

        print("pvec shape: " + str(pvec.shape))
        print("pvec: " + str(pvec[:, :5]))

        # Rotmat = np.array([[1,0,0,0],
        #                    [0, np.cos(-np.pi/4), -np.sin(-np.pi/4), 0], 
        #                    [0, np.sin(-np.pi/4), np.cos(-np.pi/4), 0], 
        #                    [0,0,0,1]])
        
        Rotmat = np.array([[1,0,0,0],
                           [0, np.cos(np.pi/4), -np.sin(np.pi/4), 0], 
                           [0, np.sin(np.pi/4), np.cos(np.pi/4), 0], 
                           [0,0,0,1]])

        # Rotmat = np.array([[1,0,0,0],
        #                    [0, np.cos(-np.pi/2), -np.sin(-np.pi/2), 0], 
        #                    [0, np.sin(-np.pi/2), np.cos(-np.pi/2), 0], 
        #                    [0,0,0,1]])

        # W = np.linalg.inv(self.map_object.tf_pose)
        # print(Rotmat)
        # pcl = ((W @ S_inv @ Kh_inv @ pvec) *-depth_z).T[:, :3]
        # pcl = ((S_inv @ Kh_inv @ pvec) *-depth_z).T[:, :3]
        pcl = ((S_inv @ Kh_inv @ pvec) *-depth_z)
        pcl = (Rotmat @ pcl).T[:, :3]
        # pcl = ((S_inv @ np.linalg.inv(Rotmat) @ Kh_inv @ pvec) *-depth_z).T[:, :3]

        # for i in range(10):
        #     print("point cloud point " + str(i) + ":" + str(pcl[i]))

        # pcl[:,0] = -1*pcl[:,0]
        pcl[:,1] = -1*pcl[:,1]

        
        
        # temp = pcl[:,0]
        # return ((Kh_inv @ pvec) * depth_z).T[:, :3]
        # return ((S_inv @ Kh_inv @ pvec) *-depth_z).T[:, :3]
        return pcl
        
    
    def generate_pointcloud(self, rgb_frame, depth_frame, se3_pose):
        """Uses purely azimuth, elevation, and depth data from camera horizontal and vertical FOV.
        Combines with odometry/pose SE3 data to give XYZRGB pointcloud in world frame. Does not convert points to camera frame
        using camera model calculation.

        rgbd_mat: image file, hxwx3, depth: hxwx1; hxwx4
        se3_pose: SE3 matrix 4x4
        return the global point cloud

        NOTE on TartanAir Dataset Field Of View (FOV) calculation:
        Image resolution of camera: 640 x 480 pixels
        Aspect ratio: 4:3
        Assume FOV angles are divided equally between the horizontal and vertical directions; no fish-eye lens type distortion.
        Horizontal FOV = FOV angle * (4/7) = 81.87 * (4/7) = 46.8 degrees
        Vertical FOV = FOV angle * (3/7) = 81.87 * (3/7) = 35.1 degrees"""

        # Extract the RGB and depth values from the RGBD frame
        # rgb = rgbd_frame[:, :, :3]
        # depth = rgbd_frame[:, :, 3] 

        transform = T.ToPILImage()
        intrinsics_vec=[320.0, 320.0, 320.0, 240.0]
        fx, fy, cx, cy = intrinsics_vec[0], intrinsics_vec[1], intrinsics_vec[2], intrinsics_vec[3]

        K = np.array([[fx, 0, cx, 0],
              [0, fy, cy, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
        
        print("rgb shape before resize: " + str(rgb_frame.shape))
        print("depth shape before resize: " + str(depth_frame.shape))

        # resize rgb to be the same shape as depth image 
        rgb_frame = transform(rgb_frame)
        rgb_frame = rgb_frame.resize((depth_frame.shape[1], depth_frame.shape[0]), Image.Resampling.LANCZOS)
        rgb_frame = np.array(rgb_frame)

        print("rgb shape after resize: " + str(rgb_frame.shape))
        print("depth shape after resize: " + str(depth_frame.shape))

        rgb = rgb_frame
        depth = depth_frame

        # Get the height and width of the image
        height, width, _ = rgb.shape
        # height, width = depth.shape

        # Create arrays of pixel coordinates
        # x, y = np.meshgrid(np.arange(width), np.arange(height))
        x, y = np.arange(width), np.arange(height)


        # Reshape the pixel coordinates and depth into vectors
        x = x.reshape(-1)
        y = y.reshape(-1)

        print("x shape: " + str(x.shape))
        print("x: " + str(x[:10]))
        print("y shape: " + str(y.shape))
        print("depth shape: " + str(depth.shape))

        # Get the intrinsic parameters of the camera
         # Get the intrinsic matrix of the camera
        # Define the intrinsic parameters of the camera
        # Camera sensor lens aperture: 14.11 mm
        f = 4  # focal length mm
        cx = 320  # principal point x-coordinate
        cy = 240  # principal point y-coordinate
        hfov = np.radians(46.8)
        vfov = np.radians(35.1)

        # Calculate the azimuth and elevation angles for each point
        # centroid = np.array([(width - 1) / 2, (height - 1) / 2])
        centroid = np.array([cx, cy])
        dx = x - centroid[0]
        dy = y - centroid[1]

        # print(dx[:10])

        azimuth = (dx / dx.shape[0]) * hfov # in rads
        elevation = (dy / dy.shape[0]) * vfov

        azimuth, elevation = np.meshgrid(azimuth, elevation)
        elevation *= -1

        print(azimuth.shape)
        print(azimuth[:10, :10])
        print(elevation.shape)
        print(elevation[:10, :10])

        # Convert the azimuth, elevation, and depth to XYZ coordinates
        # X = depth * np.sin(azimuth) * np.cos(elevation)
        # Y = depth * np.cos(azimuth) * np.cos(elevation)
        # Z = depth * np.sin(elevation)
        X = np.multiply(np.multiply(depth, np.sin(azimuth)), np.cos(elevation))
        Y = np.multiply(np.multiply(depth, np.cos(azimuth)), np.cos(elevation))
        Z = np.multiply(depth, np.sin(elevation))

        # reshape stuff 
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)
        
        # Convert the XYZ coordinates to homogeneous coordinates
        P = np.vstack((X, Y, Z, np.ones_like(Z)))
        # print(P[:, :5])

        # Transform the 3D points to the world frame
        P_world = se3_pose @ P
        print("SE3 pose: " + str(se3_pose))
        print("world coord vector: " + str(P_world[:, :5]))

        # Extract the XYZ coordinates and color information from the point cloud
        X_world = P_world[0, :]
        # print("X world shape: " + str(X_world.shape))
        Y_world = P_world[1, :]
        Z_world = P_world[2, :]
        R = rgb[:,:,0].reshape(-1) / 255.0  #?do we need to divide by 255, what color space convention are we using
        G = rgb[:,:,1].reshape(-1) / 255.0
        B = rgb[:,:,2].reshape(-1) / 255.0

        # Stack the XYZ and color information into a single array
        pointcloud = np.vstack((X_world, Y_world, Z_world, R, G, B)).transpose()

        return pointcloud

    def track(self, tstamp, image, depth, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            """ SLAM"""
            # check there is enough motion
            self.filterx.track(tstamp, image, None, intrinsics)

            # local bundle adjustment
            self.frontend()

            """######################Semantic Mapping Additions######################"""
            # get current slam frontend timestep 
            curr_tstamp = self.frontend.t1     

            # update map if the slam system's timestep actually changes 
            if curr_tstamp != self.last_timestamp:
                # add new pose to mapper 
                pose = SE3(self.video.poses[curr_tstamp]).matrix().cpu()
                self.map_object.propagate(pose)

                # pcl_xyz_rgb = self.generate_pinhole_pointcloud(image[0], depth[0], pose)
                # pcl_xyz = self.generate_pointcloud(image[0], depth[0], pose)[:,:3]
                print("PRINTING DEPTH!!!!!!!!!!!!!!!!")
                print(depth[0])

                print("tf pose: " + str(self.map_object.tf_pose))
                if self.map_object.tf_pose is not None:
                    pcl_xyz = self.generate_pinhole_pointcloud(image[0], depth[0], pose)
                    labels, _ = self.map_object.label_points(pcl_xyz)
                    labels = np.expand_dims(labels.cpu().numpy(), axis=1)

                    img = Image.fromarray(depth[0].cpu().numpy())
                    img = img.convert('RGB')
                    img.save("depth_left_" + str(self.i) + ".png")
                    self.i += 1

                    # # print("labels shape: " + str(labels.shape))
                    # # print(labels[:10])

                    labeled_pc = np.hstack((pcl_xyz, labels))
                    labeled_pc_torch = torch.from_numpy(labeled_pc).to(device="cuda", non_blocking=True)
                    self.map_object.update_map(labeled_pc_torch)

                # Add points to the map for the right image 
                # pcl_xyz = self.generate_pinhole_pointcloud(image[1], depth[1], pose)
                # labels, _ = self.map_object.label_points(pcl_xyz)
                # labels = np.expand_dims(labels.cpu().numpy(), axis=1)

                # img = Image.fromarray(depth[1].cpu().numpy())
                # img = img.convert('RGB')
                # img.save("depth_right_" + str(self.i) + ".png")

                # labeled_pc = np.hstack((pcl_xyz, labels))
                # labeled_pc_torch = torch.from_numpy(labeled_pc).to(device="cuda", non_blocking=True)
                # self.map_object.update_map(labeled_pc_torch)

            # new last timestep
            self.last_timestamp = curr_tstamp
            """#####################################################################"""
        # if self.i >= 12:
        #     return True 
        return False

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

