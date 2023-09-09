import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import json
import argparse
import glob
from PIL import Image as PILImage
from PIL.ExifTags import GPSTAGS, TAGS
from pathlib import Path
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

IMG_EXTENSIONS = ["png", "PNG", "jpg", "JPG"]
def get_sorted_image_names_in_dir(dir_path: str):
    image_paths = []
    for extension in IMG_EXTENSIONS:
        search_path = os.path.join(dir_path, f"*.{extension}")
        image_paths.extend(glob.glob(search_path))

    return sorted(image_paths)


def __compute_sensor_width_from_exif(exif_data) -> float:
        """Compute sensor_width_mm from `ExifImageWidth` tag,

        Equation: sensor_width = pixel_x_dim / focal_plane_x_res * unit_conversion_factor

        Returns:
            sensor_width_mm.
        """

        sensor_width_mm = 0.0

        # Read `ExifImageWidth` and `FocalPlaneXResolution`.
        pixel_x_dim = exif_data.get("ExifImageWidth")
        focal_plane_x_res = exif_data.get("FocalPlaneXResolution")
        focal_plane_res_unit = exif_data.get("FocalPlaneResolutionUnit")
        if (
            pixel_x_dim is not None
            and pixel_x_dim > 0
            and focal_plane_x_res is not None
            and focal_plane_x_res > 0
            and focal_plane_res_unit is not None
            and focal_plane_res_unit > 0
        ):
            ccd_width = pixel_x_dim / focal_plane_x_res

            if focal_plane_res_unit == CENTIMETERS_FOCAL_PLANE_RES_UNIT:
                sensor_width_mm = ccd_width * 10.0  # convert cm to mm
            elif focal_plane_res_unit == INCHES_FOCAL_PLANE_RES_UNIT:
                sensor_width_mm = ccd_width * MILLIMETERS_PER_INCH  # convert inch to mm

        return sensor_width_mm
    
def get_intrinsics_from_exif(exif_data, img_w_px, img_h_px):
    """Constructs the camera intrinsics from exif tag.

    Equation: focal_px=max(w_px,h_px)∗focal_mm / ccdw_mm

    Ref:
    - https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif.html
    - https://github.com/colmap/colmap/blob/e3948b2098b73ae080b97901c3a1f9065b976a45/src/util/bitmap.cc#L282
    - https://openmvg.readthedocs.io/en/latest/software/SfM/SfMInit_ImageListing/
    - https://photo.stackexchange.com/questions/40865/how-can-i-get-the-image-sensor-dimensions-in-mm-to-get-circle-of-confusion-from # noqa: E501

    Returns:
        intrinsics matrix (3x3).
    """
    max_size = max(img_w_px, img_h_px)

    # Initialize principal point.
    center_x = img_w_px / 2
    center_y = img_h_px / 2

    # Initialize focal length as None.
    focal_length_px = None

    # Read from `FocalLengthIn35mmFilm`.
    focal_length_35_mm = exif_data.get("FocalLengthIn35mmFilm")
    if focal_length_35_mm is not None and focal_length_35_mm > 0:
        focal_length_px = focal_length_35_mm / 35.0 * max_size
    else:
        # Read from `FocalLength` mm.
        focal_length_mm = exif_data.get("FocalLength")
        if focal_length_mm is None or focal_length_mm <= 0:
            return None

        # Compute sensor width, either from database or from EXIF.
        sensor_width_mm = __compute_sensor_width_from_exif(exif_data)
        if sensor_width_mm > 0.0:
            focal_length_px = focal_length_mm / sensor_width_mm * max_size

    if focal_length_px is None or focal_length_px <= 0.0:
        return None

    return focal_length_px, focal_length_px, float(center_x), float(center_y)

def load_image(img_path: str, return_intrinsics=False):
    """Load the image from disk.

    Notes: EXIF is read as a map from (tag_id, value) where tag_id is an integer.
    In order to extract human-readable names, we use the lookup table TAGS or GPSTAGS.
    Images will be converted to RGB if in a different format.

    Args:
        img_path (str): the path of image to load.

    Returns:
        loaded image in RGB format.
    """
    original_image = PILImage.open(img_path)

    exif_data = original_image._getexif()
    if exif_data is not None:
        parsed_data = {}
        for tag_id, value in exif_data.items():
            # extract the human readable tag name
            if tag_id in TAGS:
                tag_name = TAGS.get(tag_id)
            elif tag_id in GPSTAGS:
                tag_name = GPSTAGS.get(tag_id)
            else:
                tag_name = tag_id
            parsed_data[tag_name] = value

        exif_data = parsed_data
        
    img_fname = Path(img_path).name
    image_array = np.array(original_image)
    bgr_image_array = image_array[:, :, ::-1]
    if return_intrinsics:
        return bgr_image_array, get_intrinsics_from_exif(exif_data, original_image.size[0], original_image.size[1])
    else:
        return bgr_image_array

def image_stream(data, stride, use_depth=False):
    """ image generator """
    all_image_paths = get_sorted_image_names_in_dir(data)
    num_all_imgs = len(all_image_paths)
    _, (fx, fy, cx, cy) = load_image(all_image_paths[0], return_intrinsics=True)
    
    for t, imfile in enumerate(all_image_paths):
        image = load_image(imfile)
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
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
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
    for (t, image, intrinsics, depth) in tqdm(image_stream(args.data, args.stride, args.use_depth)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, depth, intrinsics=intrinsics)

    if args.output_dir is not None:
        save_reconstruction(droid, args.output_dir)

