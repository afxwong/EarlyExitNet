# LEENet: Learning Optimal Early Exit Policy for Efficiency Improvements in DNNs

Well-trained deep neural networks (DNNs) treat all test samples equally during prediction. Adaptive DNN inference with early exiting leverages the observation that some test examples can be easier to predict than others. This repository presents LEENet, a brand-new approach to early exit neural DNN models. Instead of traditional early exit networks where gate thesholds are either hardcoded or learned via inference budget thresholds, LEENet learns to optimize a cost/accuracy tradeoff balance based on a single alpha hyperparameter set by the user. This allows the model to choose an appropriate risk level for certain images which the model is less confident at earlier exit indices. Initial experiments are being conducted via DenseNet121, ResNet50, and VGG11 networks for computer vision tasks (ImageNetTE, CIFAR-10, CIFAR-100). Our initial results demonstrate that by solely tuning alpha, LEENet networks can vastly outperform traditional DNNs and also outperforms other early exit techniques in certain circumstances.

## Setup
Any python version above 3.8 will suffice for this project. If this is a CUDA-enabled PC, install your latest version of CUDA software and the appropriate PyTorch software. All other packages are in the requirements.txt file linked in the repository.

## Usage
All appropriate model architectures are either hardcoded in this repository, or are downloaded via public channels. Pretrained transfer learning models can be found [here](https://drive.google.com/drive/folders/1fzTctAh_UhlHtxb_INw86GjkU3ls1_LR?usp=sharing). Any model not in this list is already trained on the proper dataset and does not require an initial .pth file.

#### Train a multi-exit model:
```
python train.py --arch <model_architecture> --data <dataset_name> --alpha <alpha_value>
```

For example, training a DenseNet121 model on CIFAR-10 with an alpha value of 0.5 would be:
```
python train.py --arch densenet121 --data cifar10 --alpha 0.5
```

For example, training a ResNet50 model on CIFAR-100 with an alpha value of 0.5 would be:
```
python train.py --arch resnet50 --data cifar100 --alpha 0.5
```

For example, training a VGG11 model on ImageNetTE with an alpha value of 0.5 would be:
```
python train.py --arch vgg11 --data imagenette --alpha 0.5
```
Please see [args.py](args.py) for a full list of arguments which can be passed in to control the training porcess.
