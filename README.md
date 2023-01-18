# **MultiAct: Long-Term 3D Human Motion Generation from Multiple Actions**

<p align="center">  
<img src="./assets/qualitative results.png">  
</p> 

## Introduction
This repo is official **[PyTorch](https://pytorch.org)** implementation of **[**MultiAct: Long-Term 3D Human Motion Generation from Multiple Actions** (AAAI 2023 Oral.)](https://arxiv.org/abs/2212.05897v2)**.

## Quick demo
* Install **[PyTorch](https://pytorch.org)** and Python >= 3.8.13. Run `sh requirements.sh` to install the python packages. You should slightly change `torchgeometry` kernel code following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).
* Download the pre-trained model from [here](https://drive.google.com/file/d/1opOAjbExu1v8_frMOST7SZV7PMD6SP_G/view?usp=share_link) and unzip in `${ROOT}/output`.
* Prepare BABEL dataset following [here](https://github.com/TaeryungLee/MultiAct_RELEASE#babel-dataset).
* Prepare SMPL-H body model following [here](https://github.com/TaeryungLee/MultiAct_RELEASE#smplh-body-model).
* Run `python generate.py --env gen --gpu 0 --mode gen_short` for the short-term generation.
* Run `python generate.py --env gen --gpu 0 --mode gen_long` for the long-term generation.
* Generated motions are stored in `${ROOT}/output/gen_release/vis/`.


<!-- ## Directory  
### Root  
The `${ROOT}` is described as below.  
```  
${ROOT}  
|-- data  
|-- demo
|-- common  
|-- main  
|-- output  
```  
* `data` contains data loading codes and soft links to images and annotations directories.  
* `demo` contains demo codes.
* `common` contains kernel codes for I2L-MeshNet.  
* `main` contains high-level codes for training or testing the network.  
* `output` contains log, trained models, visualized outputs, and test result.   -->


## Preparation
### BABEL dataset
* Prepare BABEL dataset following [here](https://babel.is.tue.mpg.de).
* Unzip AMASS and babel_v1.0_release folder in dataset directory as below.
```  
${ROOT}  
|-- dataset
|   |-- BABEL
|   |   |-- AMASS
|   |   |-- babel_v1.0_release
```

### SMPLH body model
* Prepare SMPL-H body model from [here](https://mano.is.tue.mpg.de).
* Place the human body 3D model files in human_models directory as below.
```  
${ROOT}  
|-- human_models
|   |-- SMPLH_MALE.pkl
|   |-- SMPLH_FEMALE.pkl
|   |-- SMPLH_NEUTRAL.npz
```

### Body visualizer
* We use the body visualizer code released in this [repo](https://github.com/nghorbani/body_visualizer.git).
* Running requirements.sh installs the body visualizer in `${ROOT}/body_visualizer/`.

## Running MultiAct
### Train
* Run `python train.py --env train --gpu 0`.
* Running this code will override the downloaded checkpoints.

### Test
* Run `python test.py --env test --gpu 0`.
* Note that the variation of the generation result depends on the random sampling of the latent vector from estimated prior Gaussian distribution. Thus, the evaluation result may be slightly different from the reported metric scores in our [paper](https://arxiv.org/abs/2212.05897v2).
* Evaluation result is stored in the log file in `${ROOT}/output/test_release/log/`.

### Short-term generation
* Run `python generate.py --env gen --gpu 0 --mode gen_short` for the short-term generation.
* Generated motions are stored in `${ROOT}/output/gen_release/vis/single_step_unseen`.

### Long-term generation
#### Generating long-term motion at once
* Run `python generate.py --env gen --gpu 0 --mode gen_long` for the long-term generation.
* Generated motions are stored in `${ROOT}/output/gen_release/vis/long_term/(exp_no)/(sample_no)/(step-by-step motion)`.

#### Generating long-term motion step-by-step via resuming from previous generation results
* Modify environment file `${ROOT}/envs/gen.yaml` to match your purpose.
* Mark `resume: True` in environment file.
* Specify `resume_exp, resume_sample, and resume_step` to determine which point to continue the generation.
* Generated motions are stored in `${ROOT}/output/gen_release/vis/long_term/(next_exp_no)/(sample_no)/(step-by-step motion)`.

## Reference  
```  
@InProceedings{Lee2023MultiAct,  
author = {Lee, Taeryung and Moon, Gyeongsik and Lee, Kyoung Mu},  
title = {MultiAct: Long-Term 3D Human Motion Generation from Multiple Action Labels},  
booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},  
year = {2023}  
}  
```
