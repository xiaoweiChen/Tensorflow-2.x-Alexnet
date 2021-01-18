# AlexNet-Tensorflow 2.x

使用Tensorflow 2.x实现的AlexNet。

## 配置要求

1. Python 3.x (我所使用的是 3.8.3)

2. PIL，pip install pillow或者其他方式进行安装。

3. tqdm，pip install tqdm或者其他方式进行安装。

4. TensorFlow version: 2.x(我所使用的是2.4.0+GPU，Windows 10 x64)

5. 数据集格式(可参考[image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory)API 的说明[可能需要翻墙])：

   ```
   main_directory/
   ...class_a/
   ......a_image_1.jpg
   ......a_image_2.jpg
   ...class_b/
   ......b_image_1.jpg
   ......b_image_2.jpg
   ```

## 如何使用

1. 本库提供三个脚本：

   model.py：提供网络结构，以及相关的操作定义。

   train-test.py：使用定义的网络类进行训练和/或测试的任务。

2. 本库提供了两个用于测试的数据集：

   [imagenet2012](dataset/imagenet/README.md)：经典数据集，有1000个类别。可以在相应目录下找到获取该数据集的方式。

   [flower_photos](dataset/flower_photos/README.md)：这个是tensorflow 2.x中，分类网络教程中所使用的数据集，有5个类别。可以在目录下的README中看到获取的方式。

3. 如何进行训练和测试，脚本中的信息会给予你帮助

   ```
   python train-test.py -h
   usage: train-test.py [-h] [--enable-tenorflow-verbose] [--use-whole-network-model]
                        [--pre-train-model-path-dir PRE_TRAIN_MODEL_PATH_DIR] [--test-image TEST_IMAGE] -d DATASET_PATH
                        [--classes N] [--image-height N] [--image-width N] [--opt-type OPT_TYPE] [--momentum MOMENTUM] [--lr LR]
                        [--epochs N] [--batchsize N] [--gpu-id GPU_ID]
   
   Tensorflow 2.x Alexnet Training
   
   optional arguments:
     -h, --help            show this help message and exit
     --enable-tenorflow-verbose
     --use-whole-network-model
     --pre-train-model-path-dir PRE_TRAIN_MODEL_PATH_DIR
     --test-image TEST_IMAGE
     -d DATASET_PATH, --dataset-path DATASET_PATH
     --classes N           images classes(default: 5, flower dataset)
     --image-height N      image height(default: 180, flower dataset)
     --image-width N       image width(default: 180, flower dataset)
     --opt-type OPT_TYPE   Optimizator type(default: RMSprop)[options: Adam, SGD, Adadelta, Adamax, Ftrl, Nadam, RMSprop]
     --momentum MOMENTUM   initial SGD momentum(default: 0.9)
     --lr LR               initial learning rate(default: 0.001)
     --epochs N            number of total epochs to run(default: 20)
     --batchsize N         batchsize (default: 8)
     --gpu-id GPU_ID       id(s) for CUDA_VISIBLE_DEVICES
   ```

   例如：

   `python train-test.py -d G:\xiaowei\dataset\flower_photos --epochs 20`

## 参考信息

https://github.com/calmisential/TensorFlow2.0_Image_Classification

https://github.com/mikechen66/AlexNet_TensorFlow2.0-2.2

https://www.tensorflow.org/tutorials/images/classification

https://www.tensorflow.org/tutorials/quickstart/beginner

https://www.tensorflow.org/tutorials/quickstart/advanced