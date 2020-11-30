
## Features
#### 1. Dataset
- [x] NTU RGB+D: Cross View (CV), Cross Subject (CS)
- [ ] SBU Kinect Interaction
- [ ] PKU-MMD

#### 2. Tasks
- [x] Action recognition
- [ ] Action detection

## Prerequisites
Our code is based on **Python3.5**. There are a few dependencies to run the code in the following:
- Python (>=3.5)
- PyTorch (0.4.0)
- [torchnet](https://github.com/pytorch/tnt)
- Visdom
- Other version info about some Python packages can be found in `requirements.txt`

## Usage
#### Data preparation
##### NTU RGB+D
To transform raw NTU RGB+D data into numpy array (memmap format ) by this command:
```commandline
python ./feeder/ntu_gendata.py --data_path <path for raw skeleton dataset> --out_folder <path for new dataset>
```




Please first train a model
#### Training
Before you start the training, you have to launch [visdom](https://github.com/facebookresearch/visdom) server.
```commandline
python -m visdom
```
To train the model, you should note that:
 - ```--dataset_dir``` is the **parents path** for **all** the datasets,
 - ``` --num ``` the number of experiments trials (type: list).
```commandline
python main.py --mode train --model_name HCN --dataset_name NTU-RGB-D-CV --num 01
```
To run a new trial with different parameters, you need to:
- Firstly, run the above training command with a new trial number, e.g, ```--num 03```, thus you will got an error.
- Secondly, copy a  parameters file from the ```./HCN/experiments/NTU-RGB-D-CV/HCN01/params.json``` to the path of your new trial ```"./HCN/experiments/NTU-RGB-D-CV/HCN03/params.json"``` and modify it as you want.
- At last, run the above training command again, it will works.

#### Testing
```commandline
python main.py --mode test --load True --model_name HCN --dataset_name NTU-RGB-D-CV --num 01

#### Attack
python admm_attack.py --mode test --load True --model_name HCN --dataset_name NTU-RGB-D-CV --num 02 --targeted False --target_label 0 --beta 1.0

#### Evaluation and Defense
python evaluate.py --mode test --load True --model_name HCN --dataset_name NTU-RGB-D-CV --num 02 --sigma 0.1 --targeted False --target_label 0 --beta 1.0 --apply_defense False

#### Certification
python certify.py --mode test --load True --model_name HCN --dataset_name NTU-RGB-D-CV --num 03 --sigma 0.1

```



#### Load and Training
You also can load a half trained model, and start training it from a specific checkpoint by the following command:
```commandline
python main.py --dataset_dir <parents path for all the datasets> --mode load_train --load True --model_name HCN --dataset_name NTU-RGB-D-CV --num 01 --load_model <path for  trained model>
```
