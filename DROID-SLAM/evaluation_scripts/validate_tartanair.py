import sys
sys.path.append('droid_slam')
sys.path.append('Data')
sys.path.append('thirdparty/tartanair_tools')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob
import time
import yaml
import argparse
from TartanAir import TartanAirDataset
from torch.utils.data import DataLoader
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32
from std_msgs.msg import ColorRGBA

from droid import Droid

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[320.0, 320.0, 320.0, 240.0], stereo=False):
    """ image generator """

    # read all png images in folder
    ht0, wd0 = [480, 640]
    images_left = sorted(glob.glob(os.path.join(datapath, 'image_left/*.png')))
    images_right = sorted(glob.glob(os.path.join(datapath, 'image_right/*.png')))

    data = []
    for t in range(len(images_left)):
        images = [ cv2.resize(cv2.imread(images_left[t]), (image_size[1], image_size[0])) ]

        # add in right image because stereo
        if stereo:
            images += [ cv2.resize(cv2.imread(images_right[t]), (image_size[1], image_size[0])) ]
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)

        intrinsics = .8 * torch.as_tensor(intrinsics_vec)

        data.append((t, images, intrinsics))

    return data

# Remap colors to np array 0 to 1
def remap_colors(colors):
    # color
    colors_temp = np.zeros((len(colors), 3))
    for i in range(len(colors)):
        colors_temp[i, :] = colors[i]
    colors = colors_temp.astype("int")
    colors = colors / 255.0
    return colors

def publish_voxels(map_object, min_dim, max_dim, grid_dims, colors, next_map):
    if map_object.global_map is not None:
        next_map.markers.clear()
        marker = Marker()
        marker.id = 0
        marker.ns = "Global_Semantic_Map"
        marker.header.frame_id = "map" # change this to match model + scene name LMSC_000001
        marker.type = marker.CUBE_LIST
        marker.action = marker.ADD
        marker.lifetime.secs = 0
        marker.header.stamp = rospy.Time.now()

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1

        marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
        marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
        marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

        print(map_object.global_map.shape)
        semantic_labels = map_object.global_map[:,3:]

        print("semantic labels shape: " + str(semantic_labels.shape))

        centroids = map_object.global_map[:, :3]

        # Threshold here
        total_probs = np.sum(semantic_labels, axis=-1, keepdims=False)

        print("total probs: " + str(total_probs.shape))

        not_prior = total_probs > 1

        # print("not prior: " + str(not_prior))

        # semantic_labels = semantic_labels[not_prior, :]
        # centroids = centroids[not_prior, :]

        print("not prior semantic labels: " + str(semantic_labels.shape))

        semantic_labels = np.argmax(semantic_labels, axis=-1)
        semantic_labels = semantic_labels.reshape(-1, 1)

        for i in range(semantic_labels.shape[0]):
            pred = semantic_labels[i]
            if pred != 0:
                print("pred: " + str(pred))
            point = Point32()
            color = ColorRGBA()
            point.x = centroids[i, 0]
            point.y = centroids[i, 1]
            point.z = centroids[i, 2]
            color.r, color.g, color.b = colors[pred].squeeze()

            color.a = 1.0
            marker.points.append(point)
            marker.colors.append(color)

        next_map.markers.append(marker)
    return next_map

def publish_local_map(labeled_grid, centroids, grid_params, colors, next_map):
    max_dim = grid_params["max_bound"]
    min_dim = grid_params["min_bound"]
    grid_dims = grid_params["grid_size"]

    next_map.markers.clear()
    marker = Marker()
    marker.id = 0
    marker.ns = "Local Semantic Map"
    marker.header.frame_id = "map"
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

    X, Y, Z, C = labeled_grid.shape
    semantic_labels = labeled_grid.view(-1, C).detach().cpu().numpy()
    centroids = centroids.detach().cpu().numpy()

    semantic_sums = np.sum(semantic_labels, axis=-1, keepdims=False)
    valid_mask = semantic_sums >= 1

    semantic_labels = semantic_labels[valid_mask, :]
    centroids = centroids[valid_mask, :]

    semantic_labels = np.argmax(semantic_labels / np.sum(semantic_labels, axis=-1, keepdims=True), axis=-1)
    semantic_labels = semantic_labels.reshape(-1, 1)

    for i in range(semantic_labels.shape[0]):
        pred = semantic_labels[i]
        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]
        color.r, color.g, color.b = colors[pred].squeeze()

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)

    next_map.markers.append(marker)
    return next_map

def evaluation(droid, dataloader_tartan, scenedir, visualize, map_method, map_pub, next_map):
    # loop thru data and track each image 
    for (tstamp, image, depth, intrinsics) in tqdm(dataloader_tartan):
        droid.track(tstamp[0], image[0], depth[0], intrinsics=intrinsics[0])

        if visualize:
            if rospy.is_shutdown():
                exit("Closing Python")
            try:
                if map_method == "global" or map_method == "local":
                    # print("Got to right before publish voxels")
                    map = publish_voxels(droid.map_object, droid.grid_params['min_bound'], droid.grid_params['max_bound'], droid.grid_params['grid_size'], colors, next_map)
                    # print(map)
                    map_pub.publish(map)
                    # print("Got to here after map_pub.publish")
                elif map_method == "local":
                    map = publish_local_map(droid.map_object.local_map, droid.map_object.centroids, droid.grid_params, colors, next_map)
                    map_pub.publish(map)
            except Exception as e:
                exit("Publishing broke: " + str(e))

    # fill in non-keyframe poses + global BA
    traj_est = droid.terminate(image_stream(scenedir))

    ### do evaluation ###
    evaluator = TartanAirEvaluator()
    gt_file = os.path.join(scenedir, "pose_left.txt")
    traj_ref = np.loadtxt(gt_file, delimiter=' ')[:, [1, 2, 0, 4, 5, 3, 6]] # ned -> xyz

    # usually stereo should not be scale corrected, but we are comparing monocular and stereo here
    results = evaluator.evaluate_one_trajectory(
        traj_ref, traj_est, scale=True, title=scenedir[-20:].replace('/', '_'))
    
    print(results)
    ate_list.append(results["ate_score"])

    return ate_list

if __name__ == '__main__':
    # run on the gpu
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="datasets/TartanAir")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)

    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=15)
    parser.add_argument("--frontend_window", type=int, default=20)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--upsample", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=20.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    # Model Parameters for Neural BKI
    MODEL_NAME = "ConvBKI_Single_02_odom"
    model_params_file = os.path.join(os.getcwd(), "NeuralBKI_Config", MODEL_NAME + ".yaml")
    with open(model_params_file, "r") as stream:
        try:
            model_params = yaml.safe_load(stream)
            dataset = model_params["dataset"]
        except yaml.YAMLError as exc:
            print(exc)

    # Data Parameters for Neural BKI
    data_params_file = os.path.join(os.getcwd(), "NeuralBKI_Config", dataset + ".yaml")
    with open(data_params_file, "r") as stream:
        try:
            data_params = yaml.safe_load(stream)
            NUM_CLASSES = data_params["num_classes"]
            colors = remap_colors(data_params["colors"])
            DATA_DIR = data_params["data_dir"]
            ignore_labels = data_params["ignore_labels"]

        except yaml.YAMLError as exc:
            print(exc)

    # CONSTANTS
    SEED = model_params["seed"]
    NUM_FRAMES = model_params["num_frames"]
    MODEL_RUN_DIR = os.path.join("Models", "Runs", MODEL_NAME + "_" + dataset)
    NUM_WORKERS = model_params["num_workers"]
    FLOAT_TYPE = torch.float32
    LABEL_TYPE = torch.uint8
    MAP_METHOD = model_params["map_method"]
    LOAD_EPOCH = model_params["load_epoch"]
    LOAD_DIR = model_params["save_dir"]
    VISUALIZE = model_params["visualize"]
    MEAS_RESULT = model_params["meas_result"]
    GEN_PREDS = model_params["gen_preds"]
    FROM_CONT = model_params["from_continuous"]
    TO_CONT = model_params["to_continuous"]
    PRED_PATH = model_params["pred_path"]

    from data_readers.tartan import test_split
    from evaluation.tartanair_evaluator import TartanAirEvaluator

    if not os.path.isdir("figures"):
        os.mkdir("figures")

    ate_list = []

    print("Performing evaluation on {}".format(args.datapath))

    torch.cuda.empty_cache()
    droid = Droid(args, model_params, NUM_CLASSES, ignore_labels)
    
    # create TartanAir dataset 
    test_ds = TartanAirDataset(directory=args.datapath, device=device) 

    # create TartanAir dataloader
    dataloader_tartan = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=NUM_WORKERS, pin_memory=True)

    map_pub = None
    if VISUALIZE:
        rospy.init_node('talker', anonymous=True)
        map_pub = rospy.Publisher('SemMap_global', MarkerArray, queue_size=10)
        next_map = MarkerArray()
    # perform evaluation on the data 
    ate_list = evaluation(droid, dataloader_tartan, args.datapath, VISUALIZE, MAP_METHOD, map_pub, next_map)

    print("Results")
    print(ate_list)

    if args.plot_curve:
        import matplotlib.pyplot as plt
        ate = np.array(ate_list)
        xs = np.linspace(0.0, 1.0, 512)
        ys = [np.count_nonzero(ate < t) / ate.shape[0] for t in xs]

        plt.plot(xs, ys)
        plt.xlabel("ATE [m]")
        plt.ylabel("% runs")
        plt.show()

