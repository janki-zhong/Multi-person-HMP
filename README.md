# Multi-person-HMP
Multi-person HMP-xinyi

# FBINet model
Title：[Fine-Grained Behavior Interaction-Aware Network for Efficient Multi-Person Motion Forecasting](https://link.springer.com/article/10.1007/s00530-025-02199-1). In Multimedia Systems 2026.

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

The relevant parameter settings are all in the /utils/config_UMPM.py file.

## Test
We provide the evaluation code on the Mocap_UMPM dataset, you can run 
`python test.py`.

On the Mocap_UMPM dataset, batch_size=30;
On the mupots dataset, batch_size=16;
On the Mix1/Mix2 dataset, batch_size=10.

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
├── train_short.py
├── test_short.py
├── train_long.py
├── test_long.py
├── train_FBINet.py
├── test_ablation.py
```

## Train
For training on Mocap_UMPM dataset, you can run `python train_short.py` for short-term prediction.
You can run `python train_long.py` for long-term prediction.

## Test
We provide the evaluation code on the Mocap_UMPM dataset, you can run 
`python test_short.py` for short-term prediction.
You can run `python test_long.py` for long-term prediction.


