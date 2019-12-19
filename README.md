# TransMatch
PyTorch implementation of TransMatch: a transfer learning scheme for semi-supervised few-shot learning. 
The following code is for miniImagenet.

#### Summary

* [Requirements](#environment-requirements)
* [Download data and pre-trained model](#project-architecture)
* [Run experiments on miniImagenet](#running-experiments)
* [Performance](#performance)


## Requirements
Use conda
```bash
conda env create -f environment.xml
```
Use pip
```
pip install -r requirements.txt
```

## Download data and pre-trained model
1. Download miniImageNet: [download link](https://drive.google.com/file/d/1Mmx4pi-29FOh9R2wMS1OpqxSuXEUP_vD/view?usp=sharing)

2. Download pretrained model on base-class data: [download link](https://drive.google.com/file/d/1CUluyeErZ919EVV1WeAD0QYN7d130rZm/view?usp=sharing)

Then Unzip the `MY_mini_data.zip`.

The project folder should look like:
```bash
main.py
mini_loader.py
wide_models.py
wideresnet_legacy.py
pretrained_model_on_base_class.pth.tar
utils
MY_mini_data
|   base_data
|   novel_data
└─── miniImagenet_base_novel
    └───base
    └───novel
```

## Run experiments on miniImagenet
Please check [main.py](main.py) for the details.

Notice: 
1. The corresponding results are stored in `final_result.csv` (totally 600 test results)
2. Imprinting will be stored in `imprinting_result.csv` (totally 600 test results)
3. logXXX.png is the plot for the test of 0 to XXX; each line stands for the change of test accuracy over epochs
4. Fine-tuning for each test experiment may last for 10-20 minutes, so it may take 4-6 days to finish all 600 test experiments. You could also just let the code run for 100 test experiments. The results are similar.

### 5-way 5-shot 100-unlabel
#### TransMatch
```bash
python main.py --gpu 0 --num-way 5 --num-sample 5 --unlabelnumber 100 --epoch 25 --checkpoint TransMatch_5_5_100
```
#### MixMatch
```bash
python main.py --gpu 0 --random --num-way 5 --num-sample 5 --unlabelnumber 100 --epoch 25 --checkpoint MixMatch_5_5_100
```
#### TransMatch with distractor class
```bash
python main.py --gpu 0 --num-way 5 --num-sample 5 --unlabelnumber 100 --distractor --distractor_class 2 --epoch 25 --checkpoint TransMatch_5_5_100_distractor_2
```
#### each log file will be like in this format:
```bash
Learning_Rate	Train_Loss	Train_Loss_X	Train_Loss_U	Valid_Loss	Valid_Acc
0.001000	0.908974	0.776229	0.026549	1.061434	56.000000	
0.001000	0.768066	0.660366	0.021540	0.905195	66.666667	
0.001000	0.813665	0.708609	0.021011	0.946537	65.333333	
0.001000	0.765839	0.673886	0.018390	0.896166	66.666667	
0.001000	0.705718	0.611745	0.018795	0.878929	69.333333	
...
0.001000	0.639709	0.563627	0.015217	0.803816	72.000000	
```
#### The change of test accuracy over 25 epochs for 600 experiments will be like (each line represents one experiment)
<p align="center"><img width="50%" src="example.png" /></p>

### 5-way 1-shot 100-unlabel
#### TransMatch 
```bash
python main.py --gpu 0 --num-way 5 --num-sample 1 --unlabelnumber 100 --epoch 20 --checkpoint TransMatch_5_1_100
```

#### MixMatch 
```bash
python main.py --gpu 0 --random --num-way 5 --num-sample 1 --unlabelnumber 100 --epoch 20 --checkpoint MixMatch_5_1_100
```

#### TransMatch with distractor class
```bash
python main.py --gpu 0 --num-way 5 --num-sample 1 --unlabelnumber 100 --distractor --distractor_class 2 --epoch 20 --checkpoint TransMatch_5_1_100_distractor_2
```

## Performance
### No distractor class for unlabeled images


|          (%)           | Imprinting  | MixMatch  | TranMatch (Ours) |
| ---------------------- | ------------ | ------------ | ------------ |
| 1-shot            | `58.68 ± 0.81` | `52.00 ± 1.20` | `63.02 ± 1.07` |
| 5-shot           | `76.06 ± 0.59` | `79.97 ± 0.62` | `81.19 ± 0.59` |

### With distractor classes for unlabeled images

|          (%)           | Imprinting  | MixMatch  | TranMatch (Ours) |
| ---------------------- | ------------ | ------------ | ------------ |
| 1-shot            | `58.68 ± 0.81` | `50.68 ± 1.15` | `60.41 ± 1.02` |
| 5-shot           | `76.06 ± 0.59` | `78.07 ± 0.69` | `79.48 ± 0.64` |

