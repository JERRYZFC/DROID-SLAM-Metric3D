import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import json
import argparse
from evo.core import sync

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation

from droid import Droid
from scipy.spatial.transform import Rotation

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    pose = np.eye(4)
    pose[:3,:3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(data, use_depth=False, stride=0):
    """ image generator """
    # Read json from {data}/transforms.json
    with open(os.path.join(data, 'transforms.json'), 'r') as f:
        transforms = json.load(f)
    fx, fy = transforms['fl_x'], transforms['fl_y']
    cx, cy = transforms['cx'], transforms['cy']
    
    image_list = []
    for _frame in transforms['frames']:
        image_list.append(_frame['file_path'])
    
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(data, imfile))
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        if use_depth:
            depth = depth_model.infer_pil(image)

        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if use_depth:
            depth = torch.as_tensor(depth)
            yield t, image[None], intrinsics, depth
        else:
            yield t, image[None], intrinsics, None

def save_reconstruction(droid, output_dir):
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy() # [N, 3, 384, 512]
    disps = droid.video.disps_up[:t].cpu().numpy() # [N, 384, 512]
    poses = droid.video.poses[:t].cpu().numpy() # [N, 7]
    intrinsics = droid.video.intrinsics[:t].cpu().numpy() # [N, 4]
    
    # transfer [fx fy cx cy] to 4x4 intrinsics matrix
    intrinsics = intrinsics[0] # [fx fy cx cy]
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[0, 0] = intrinsics[0]  # fx
    intrinsics_4x4[1, 1] = intrinsics[1]  # fy
    intrinsics_4x4[0, 2] = intrinsics[2]  # cx
    intrinsics_4x4[1, 2] = intrinsics[3]  # cy

    os.makedirs(f'{output_dir}/pose', exist_ok=True)
    os.makedirs(f'{output_dir}/color', exist_ok=True)
    os.makedirs(f'{output_dir}/depth', exist_ok=True)
    os.makedirs(f'{output_dir}/intrinsic', exist_ok=True)

    transformed_data = {
        "w": int(images[0].shape[2]),
        "h": int(images[0].shape[1]),
        "fl_x": float(intrinsics[0]) * 8.0,
        "fl_y": float(intrinsics[1]) * 8.0,
        "cx": float(intrinsics[2]) * 8.0,
        "cy": float(intrinsics[3]) * 8.0,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "camera_model": "OPENCV",
        "frames": []
    }

    # save images, poses, intrinsics
    for _id  in tqdm(range(t)):
        pose = np.linalg.inv(pose_matrix_from_quaternion(poses[_id]))
        pose[:, 2] *= -1
        pose[:, 1] *= -1
        image = images[_id]
        image = image.transpose(1, 2, 0)
        depth = (disps[_id] * 1000).astype(np.uint16)
        np.savetxt(f'{output_dir}/pose/{_id}.txt', pose)
        cv2.imwrite(f'{output_dir}/color/{_id}.jpg', image)
        cv2.imwrite(f'{output_dir}/depth/{_id}.png', depth)
        np.savetxt(f'{output_dir}/intrinsic/intrinsic_color.txt', intrinsics_4x4)
        np.savetxt(f'{output_dir}/intrinsic/intrinsic_depth.txt', intrinsics_4x4)

        
        transformed_frame = {
            "file_path": f"color/{_id}.png",
            "transform_matrix": pose.tolist(),
        }
        transformed_data['frames'].append(transformed_frame)
        output_data_path = os.path.join(output_dir, 'transforms.json')
        with open(output_data_path, 'w') as f:
            f.write(json.dumps(transformed_data, indent=4))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data directory")
    parser.add_argument("--output_dir", help="path to saved reconstruction")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=0.1, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=1.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=50, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--use_depth", action="store_true")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    if args.use_depth:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        depth_model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

    droid = None

    # need high resolution depths
    if args.output_dir is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics, depth) in tqdm(image_stream(args.data, args.use_depth, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, depth, intrinsics=intrinsics)
        tstamps.append(t)

    
    if args.output_dir is not None:
        save_reconstruction(droid, args.output_dir)