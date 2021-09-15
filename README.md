# Faster R-CNN for KITTI and BDD100K

This project is based on facebookresearch's [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
Now, the code could only run on GTX1080ti and GTX2080ti. (TITAN X may not work.)
## Installation

Connect the GPU before you start the installation
```
conda create --name py37 python=3.7
conda activate py37
conda install -c anaconda numpy
conda install -c anaconda cython
conda install ipython
pip install tqdm yacs
mkdir ~/github
cd ~/github
export INSTALL_DIR=$PWD
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ~/github
#put this repo inside ~/github
cd faster-rcnn-KITTI-BDD100k
conda install -c pytorch pytorch torchvision cudatoolkit=9.2
conda install -c psi4 gcc-5
python setup.py build develop
unset INSTALL_DIR
pip install tensorboardX
```

## Data Preparation

### BDD100K
Prepare the dataset with the below structure:
```
bdd100k/
  images/
    100k/
      train/
      val/
      test/
  labels/
    bdd100k_labels_images_train.json
    bdd100k_labels_images_val.json
```

Finally, create a symlink for the datasets:
```
ln -s {parent directory of kitti and bdd100k} {root directory of this repo}/datasets
```

## Training

### BDD100K
```
python tools/train_net.py --config-file "configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" --use_tensorboard MODEL.ROI_BOX_HEAD.NUM_CLASSES 11 DATASETS.TRAIN '("bdd100k_train",)' OUTPUT_DIR out_bdd SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0025 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" TEST.IMS_PER_BATCH 1 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000
```

For the BDD100K dataset, you might need to clip the gradient for training stability. For other hyperparameters, refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).


## Inference

Below commands loads saved weights and save the detections in the [BDD format](https://github.com/ucbdrive/bdd-data/blob/master/doc/format.md) in ``{OUTPUT_DIR}/detections``.

### BDD100K
```
python tools/test_net.py --weights out_bdd/model_0175000.pth --config-file "configs/e2e_faster_rcnn_R_101_FPN_1x.yaml" DATASETS.TEST '("bdd100k_val",)' OUTPUT_DIR out_bdd MODEL.ROI_BOX_HEAD.NUM_CLASSES 11
```

## Evaluate
To [evaluate](https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/evaluate.py) the saved detection, run the below command:
```
python tools/evaluate.py --result out_bdd/detections DATASETS.TEST '("bdd100k_val",)'
```

