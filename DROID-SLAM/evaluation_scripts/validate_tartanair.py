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

def evaluation(droid, dataloader_tartan, scenedir):
    # loop thru data and track each image 
    for (tstamp, image, depth, intrinsics) in tqdm(dataloader_tartan):
        droid.track(tstamp[0], image[0], depth[0], intrinsics=intrinsics[0])

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

    # perform evaluation on the data 
    ate_list = evaluation(droid, dataloader_tartan, args.datapath)

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

