# DroidSLAM x Metric3D


<center><img src="misc/0028.gif" width="640" style="center"></center>

# Install 
We offer two ways to setup the environment:
1. [WIP] We provide prebuilt Docker images for runing DROID-SLAM + Metric3D pipeline

2. The conda enviroment for BakedSDF. Install the dependencies and activate the environment `bakedsdf` with
```
git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git
cd DROID-SLAM
conda env create -f environment.yaml
pip install evo --upgrade --no-binary evo
pip install gdown
python setup.py install
```

# How to use
To run SfM with a dataset with only an image directory and EXIF, with image file names ending with "jpg", please create the following file structure like
```
└── {DATASET_DIR}
       ├── images
               ├── image1.jpg
               ├── image2.jpg
               ├── image3.jpg
```
and run
```
python demo.py --data ${DATASET_DIR} --output_dir ${OUPUT_DIR}
```

If you want to use Metric3D depths, you could run
```
python demo.py --data ${DATASET_DIR} --output_dir ${OUPUT_DIR} --use_depth
```

The results will save in a nerfstudio and instant-ngp data format
```
└── {OUPUT_DIR}
  ├── images
      ├── image1.jpg
      ├── image2.jpg
      ├── image3.jpg
  ├── transforms.json
```
## Acknowledgements
The code is based on
```
@article{teed2021droid,
  title={{DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras}},
  author={Teed, Zachary and Deng, Jia},
  journal={Advances in neural information processing systems},
  year={2021}
}
```
```

```