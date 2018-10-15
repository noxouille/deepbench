# Device Agnostic Deep Learning Benchmark

![NGC](https://img.shields.io/badge/ngc-18.09--py3-%2374b71b.svg)
![CUDA](https://img.shields.io/badge/cuda-10.0-%2374b71b.svg)
![cuDNN](https://img.shields.io/badge/cudnn-7.3.0-%2374b71b.svg)
![NVIDIA Driver](https://img.shields.io/badge/nvidia%20driver-410.57-%2374b71b.svg)
![Pytorch](https://img.shields.io/badge/pytorch-0.4.1%2B-%23ee4c2c.svg)
![TF](https://img.shields.io/badge/tensorflow-1.10.0-orange.svg)
![Caffe2](https://img.shields.io/badge/caffe2-0.8.1-%2325376b.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-lightgrey.svg)

This repo contains device agnostic codes of framework-wise benchmark (adapted from [u93kun](https://github.com/u39kun/deep-learning-benchmark)) and also layer-wise plus model-wise benchmark (adapted from [avik-pal](https://github.com/avik-pal/DeepLearningBenchmarks)).
Few other results were added based on my own test results. 

The scripts for layer-wise and model-wise are Pytorch based, framework-wise includes Pytorch, Caffe2, and TensorFlow.
Performance of CPUs and GPUs are compared, including the effect of adjusting the floating point precision (the new Volta architecture allows performance boost by utilizing half/mixed-precision calculations.)

# Table of Contents
<!-- TOC -->

- [How to run](#how-to-run)
    - [Layer-wise benchmark](#layer-wise-benchmark)
    - [Model-wise benchmark](#model-wise-benchmark)
    - [Framework-wise benchmark](#framework-wise-benchmark)
- [Docker support](#docker-support)
- [Visualized Results](#visualized-results)
- [Contributors](#contributors)

<!-- /TOC -->

## How to run

By default, it should run on GPU.
It will run on CPU either when GPU is not detected, or 
you manually remove `'cuda:0' if torch.cuda.is_available() else ` from the following line

```
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

### Layer-wise benchmark

```
python3 layer_benchmark.py
```

Tested layers:
* Conv3x3, stride 1, padding 1
* Conv5x5, stride 1, padding 2
* Conv3x3, stride 2, padding 1
* Conv5x5, stride 2, padding 2
* Maxpool, stride 2, padding 1
* Meanpool, stride 2, padding 1
* Batchnorm
* Dense

### Model-wise benchmark

```
python3 model_benchmark.py
```

The following models were tested with TensorFlow and Caffe2:
* vgg16
* resnet152

The following models are available to test with Pytorch:
* vgg16 
* vgg16_bn
* vgg19
* vgg19_bn
* resnet18
* resnet34
* resnet50
* resnet101
* resnet152
* Densenet161

### Framework-wise benchmark

```
python3 framework_benchmark.py -f <framework_name>
```

Available frameworks to test:
* pytorch
* tensorFlow (GPU only)
* caffe2 (GPU only)

P.S.: for some reason, with Zotac 1080Ti, caffe2 seems to have "out of memory" error for fp16 benchmark. It wasn't the case With GV100 and P5000 from NVIDIA.

## Docker support

The following command will create a result subdirectory (if it doesn't exist), and run all specified benchmarks by default.

```
./run_all_benchmark_docker.sh <device_name>
```

## Visualized Results

The results are now visualized [here](https://noxouille.github.io/tech/2018/07/22/gpu-benchmark/).
