import torch
import lietorch
from lietorch import SE3
import numpy as np

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

        # visualizer
        if not self.disable_vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        grid_params = model_params["test"]["grid_params"]
        LOAD_EPOCH = model_params["load_epoch"]
        LOAD_DIR = model_params["save_dir"]

        # create semantic mapper 
        self.map_object = GlobalMap(
            torch.tensor([int(p) for p in grid_params['grid_size']], dtype=torch.long).to(device),  # Grid size
            torch.tensor(grid_params['min_bound']).to(device),  # Lower bound
            torch.tensor(grid_params['max_bound']).to(device),  # Upper bound
            torch.load(os.path.join("/opt/Volume-DROID/DROID-SLAM/droid_slam/Models", "Weights", LOAD_DIR, "filters" + str(LOAD_EPOCH) + ".pt")), # Filters
            model_params["filter_size"], # Filter size
            num_classes=NUM_CLASSES,
            ignore_labels = ignore_labels, # Classes
            device=device # Device
        )

        # track the last timestamp 
        self.last_timestamp = 0


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

    def get_global_pointcloud(rgbd_mat, pose):
        pass
    # rgbd_mat: image file, hxwx3, depth: hxwx3
    # pose: SE3 matrix 4x4
    # return the global point cloud

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

                # # Add points to map
                # labeled_pc = np.hstack((points, pred_labels))
                # labeled_pc_torch = torch.from_numpy(labeled_pc).to(device=device, non_blocking=True)
                # map_object.update_map(labeled_pc_torch)

                # new last timestep
                self.last_timestamp = curr_tstamp
            """#####################################################################"""


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

