# Multi-person-HMP
Multi-person HMP-xinyi

# FBINet model
Title：Fine-Grained Behavior Interaction-Aware Network for Efficient Multi-Person Motion Forecasting。In Multimedia Systems 2026.

## Requirements
* python==3.10
* matplotlib==3.5.1
* numpy==1.22.3
* scipy==1.7.3
* torch==1.12.1
* transformers==4.18.0

## Get the Data
The datasets we used are all sourced from [TBIFormer](https://github.com/xiaogangpeng/tbiformer), which has made the data available online for download. Please prepare the data as follows:
```
project_folder/
├── data/
│   ├── Mocap_UMPM
│   │   ├── train_3_75_mocap_umpm.npy
│   │   ├── test_3_75_mocap_umpm.npy
│   ├── MuPoTs3D
│   │   ├── mupots_150_2persons.npy
│   │   ├── mupots_150_3persons.npy
│   ├── mix1_6persons.npy
│   ├── mix2_10persons.npy
├── dataset/
│   ├── ...
├── logs/
│   ├── ...
├── models/
│   ├── ...
├── utils/
│   ├── ...
├── train.py
├── test.py
```

## Train
For training on Mocap_UMPM dataset, you can run 
`python train.py`.

## Test
We provide the evaluation code on the Mocap_UMPM dataset, you can run 
`python test.py`.

We provide our trained model on Mocap_UMPM, you can download it from [Google Drive](https://drive.google.com/file/d/1UfQVQPFDW8PURsqRnk45loR1uMUi10iF/view?usp=sharing) and put it in logs directory.

# GCINet model
Based on the previous model, improvements have been made by incorporating geometric constraints for interactive perception reasoning, enabling the discrimination of interaction relationships in a broader sense.

## Requirements
The dependencies of the two models are the same.

## Get the Data
Directory structure:
```
project_folder/
├── data/
│   ├── Mocap_UMPM
│   │   ├── train_3_75_mocap_umpm.npy
│   │   ├── test_3_75_mocap_umpm.npy
│   ├── MuPoTs3D
│   │   ├── mupots_150_2persons.npy
│   │   ├── mupots_150_3persons.npy
│   ├── mix1_6persons.npy
│   ├── mix2_10persons.npy
├── dataset/
│   ├── ...
├── logs/
│   ├── ...
├── models/
│   ├── ...
├── utils/
│   ├── ...
├── train.py
├── test.py
```

## Train
For training on Mocap_UMPM dataset, you can run 
`python train.py`.

## Test
We provide the evaluation code on the Mocap_UMPM dataset, you can run 
`python test.py`
