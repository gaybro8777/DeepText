### DeepText: A new approach for text proposal generation and text detection in natural images.

by Zhuoyao Zhong, Lianwen Jin, Shuangping Huang, South China University of Technology (SCUT), Published in ICASSP 2017. 

### Introduction
This repository is a fork from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), and our proposed DeepText system for scene textdetection is based on the elegant framework of Faster R-CNN. 

You can refer to [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn/blob/master/README.md) and [faster-rcnn README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more information.

### Desclaimer

Please note that this repository is the demo codes (with our trained model) for DeepText system, which doesn't contain iterative regression module and linking segments and any training codes.

### Citing DeepText

If our codes are useful for your work, please cite our paper:
```
@inproceedings{icassp2017DeepText,
  title={{DeepText}: DeepText: A new approach for text proposal generation and text detection in natural images},
  author={Zhuoyao Zhong, Lianwen Jin, Shuangping Huang},
  booktitle = {International Conference on Acoustics, Speech and Signal Processing ({ICASSP})}},
  year={2017}
}
```
### Installation
1. Clone the DeepText repository
    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/zhongzhuoyao/DeepText.git
    ```

2. We'll call the directory that you cloned Faster R-CNN into `DeepText_ROOT`. Build the Cython modules
    ```Shell
    cd $DeepText_ROOT/lib
    make
    ```

3. Build Caffe and pycaffe
    ```Shell
    cd $DeepText_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html
    # For your Makefile.config:
    #   Uncomment `WITH_PYTHON_LAYER := 1`

    cp Makefile.config.example Makefile.config
    make -j8 && make pycaffe
    ```

4. Download DeepText text detection model from [one drive](https://1drv.ms/u/s!AjIwvtyYt40aadToKyWv_-wv64M), and the populate it into directory `models`. The model's name should be `vgg16_DeepText_trained_model.caffemodel`.

### How to run the demo

1. Download PASCAL VOC 2007 and 2012
-- Follow the instructions in [py-faster-rcnn README.md](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models)

2. PVANet on PASCAL VOC 2007
    ```Shell
    cd $FRCN_ROOT
    ./tools/test_net.py --net models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712.caffemodel --def models/pvanet/pva9.1/faster_rcnn_train_test_21cls.pt --cfg models/pvanet/cfgs/submit_1019.yml --gpu 0
    ```
### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

3. PVANet (compressed)
    ```Shell
    cd $FRCN_ROOT
    ./tools/test_net.py --net models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus_compressed.caffemodel --def models/pvanet/pva9.1/faster_rcnn_train_test_ft_rcnn_only_plus_comp.pt --cfg models/pvanet/cfgs/submit_1019.yml --gpu 0
    ```

### Expected results

#### Recall, Precision and F-measure on ICDAR-2013 benchmark.

| Recall (%)     | Precision (%) | F-measure (%) |
| --------- | ------- | ------- |
| *82.17*) | *87.13* | *84.58* |
