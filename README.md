# Self-Supervised Cyclic Diffeomorphic Mapping for Soft Tissue Deformation Recovery in Robotic Surgery Scenes
Implementation for TMI 2024 paper [<strong>Self-Supervised Cyclic Diffeomorphic Mapping for Soft 
Tissue Deformation Recovery in Robotic Surgery Scenes</strong>](https://ieeexplore.ieee.org/document/10630572)
by Shizhan Gong, Yonghao Long, [Kai Chen](https://ck-kai.github.io/), Jiaqi Liu,
[Yuliang Xiao](https://mikami520.github.io/), Alexis Cheng, Zerui Wang, and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/index.html).

## Sample Results
https://github.com/peterant330/SoftTissueDeformation/assets/22131710/6ba12cc4-6508-4e91-8250-86a56611534c

## Setup
We recommend to set up the environment with the following command.

`pip install -r requirements.txt `

We managed to test our code on Ubuntu 18.04 with Python 3.8 and CUDA 11.3. Our implementation is based on single GPU setting.

## Dataset
We provide a sample clip in the folder `sample_data`. Please arrange your own data as follows
```
folder/
	└── clip1/ 
	    └── img_left/
	        └── 0001.jpg
	        └── 0002.jpg
	        └── ...
	    └── img_right/
	        └── 0001.jpg
	        └── 0002.jpg
	        └── ...
	└── clip2/
	    └── img_left/
	    └── img_right/ 
	└── ...
```
where each `clip` folder corresponds to a short video clip, `img_left` are the left view of the stereo 
images, and `img_right` are the right view of the stereo images. Each image with `img_left` and  `img_right` 
represents a single frame. 

## Pre-processing
### Depth Estimation

The code for Depth estimation is adapted from [stereo-transformer](https://ck-kai.github.io/). First go to the 
`data_preprocessing/depth_estimation` folder and install the required environment.
```commandline
cd data_preprocessing/depth_estimation
pip install -r requirements.txt
```
Then type the command below for generating disparity maps

```commandline
python main.py --address path/to/store/data --model_file_name path/to/pretrained-checkpoint
```
`--address` denote the path of the datafolder and `--model_file_name` denote the path of pretrained checkpoint.
The checkpoint can be downloaded [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/ERqEFOYqdztLqWQza--4vN4B8JafIw_ECVwtV8sz8bPPMQ).

### Instrument Segmentation
First go to the 
`data_preprocessing/tool_segmentation` folder and install the required environment.
```commandline
cd data_preprocessing/tool_segmentation
conda env create -f environment.yml
conda activate CsrSeg
```
Then type the command below for generating tool segmentation masks.

```commandline
python main.py --address path/to/store/data --model_path path/to/pretrained-checkpoint
```
`--address` denote the path of the datafolder and `--model_path` denote the path of pretrained checkpoint.
Our pretrained checkpoint can be downloaded [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/EZkjabvoPitBi5QAXbEVvhsBptu_PWqIhFdID8vwsVreaA?e=dPh5SA).

## Training
 Type the command below to for training the model.
```commandline
python main.py --train_data train.pkl --eval_data eval.pkl --output_dir path/to/store/checkpoint
```
`--train_data` and `--eval_data` stores the meta information of the training and validation data. `--output_dir` stores 
the trained checkpoint.

Here is an example of `train.pkl`:

```commandline
[{'path': 'path/to/data/folder/clip1/',
  't0': '0001',
  't1': '0002',
  't2': '0003',
  't3': '0004',
  't4': '0005'},
 {'path': 'path/to/data/folder/clip1/',
  't0': '0002',
  't1': '0003',
  't2': '0004',
  't3': '0005',
  't4': '0006'},
  ...]
```

and another example of `eval.pkl`:
```commandline
[{'path': 'path/to/data/folder/clip1/',
  'sequence': ['0001',
   '0002',
   '0003',
   '0004',
   '0005',
   '0006',
   '0007',
   '0008',
   ...],},
   ...]
```
Each sequence in `--eval_data` corresponds to only 5 frame while each sequence in `eval` corresponds to a longer clip.

## Inference
 Type the command below to for model inference.
```commandline
python main.py --test_data test.pkl --model_path path/to/pretrained-checkpoint
```
`--test_data`  stores the meta information of the test data. `--model_path` denote the path of pretrained checkpoint. 
Our pretrained checkpoint can be downloaded 
[here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155187960_link_cuhk_edu_hk/EZg8W98rGbJIr0ZS3wahkWQBPNtEpL9R2yvWBY1ykTl5Hg?e=rjlhKB). 
`test.pkl` has the same format as `eval.pkl`.

## Contact
For any questions, please contact `szgong22@cse.cuhk.edu.hk`
