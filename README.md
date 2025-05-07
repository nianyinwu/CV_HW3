# NYCU Computer Vision 2025 Spring HW3
StudentID: 313551078 \
Name: 吳年茵

## Introduction
In this lab, we have to train an __instance segmentation__ model to solve an __instance segmentation task__ to segment the mask of four types of cells with 4+1 classes (including background). 

We use a dataset of 209 colored medical images for training and validation and 101 for testing. For this lab, I split the data to the ratio of 8:2 for training and validation. 
I adopted the __Mask R-CNN__ model in Pytorch as my based model, modified the number of output classes to match our task, and used ResNet's pre-trained weight. 

To improve the model's performance for this task, I also modified some backbone and added some tricks, such as a data augmentation module, to have a better metrics in AP50.

<!-- Additionally, I experiment with some tricks, such as making some __data augmentations__ or introducing the __Convolutional Block Attention Module (CBAM)__, hope to improve the model’s performance.  -->


## How to install
1. Clone this repository and navigate to folder
```shell
git clone https://github.com/nianyinwu/CV_HW3.git
cd CV_HW3
```
2. Install environment
```shell
conda env create --file hw3.yml --force
conda activate hw3
pip install scikit-image
pip install imagecodecs
```

3. Dataset
```shell
Create a folder named datas and put decompress data to this folder
Rename the test-release folder to test
```

## Split the dataset
(need to modify data path in split_data.py)
```shell
cd codes
python3 split_data.py 
```

## Generate Validation Ground Truth json file
(need to modify data path in split_data.py)
```shell
cd codes
python generate_gt_json.py 
```

## Training
```shell
cd codes
python3 train.py -e <epochs> -b <batch size> -lr <learning rate> -d <data path> -s <save path> 
```
## Testing ( Inference )
The predicted results (test-results.json) will be saved in the argument of save path .
```shell
cd codes
python3 inference.py -d <data path> -w <the path of model checkpoints> -s <save path>
```

## Performance snapshot
![image](https://github.com/nianyinwu/CV_HW3/blob/main/result/snapshot.png)
