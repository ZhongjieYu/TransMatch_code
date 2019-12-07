# TransMatch
This is the code for pytorch implementation for TransMatch on miniImagenet. (pretrained model on base-class data is attached in the link and we only show the test period for simplicity)

## Requirements
- Python 3.5
- Pytorch 3.6+
- torchvision
- pandas
- progress
- matplotlib
- numpy

## Download necassary files
download MY_mini_data: https://drive.google.com/file/d/1Mmx4pi-29FOh9R2wMS1OpqxSuXEUP_vD/view?usp=sharing
pretrained model on base-class data: https://drive.google.com/file/d/1CUluyeErZ919EVV1WeAD0QYN7d130rZm/view?usp=sharing
Then Unzip the MY_mini_data.zip 
All the files should look like this in your folder:
```
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
## Run core experiments for miniImagenet: (check main.py for detailed components for the method)
Notice: 
1. The corresponding results are stored in final_result.csv (totally 600 test results)
2. Imprinting will be stored in imprinting_result.csv (totally 600 test results)
3. logXXX.png is the plot for the test of 0 to XXX; each line stands for the change of test accuracy over epochs
### 5-way 5-shot 100-unlabel
#### TransMatch
python main.py --gpu 0 --num-way 5 --num-sample 5 --unlabelnumber 100 --epoch 25 --checkpoint TransMatch_5_5_100
#### MixMatch
python main.py --gpu 0 --random --num-way 5 --num-sample 5 --unlabelnumber 100 --epoch 25 --checkpoint MixMatch_5_5_100
#### TransMatch with distractor class
python main.py --gpu 0 --num-way 5 --num-sample 5 --unlabelnumber 100 --distractor --distractor_class 3 --epoch 25 --checkpoint TransMatch_5_5_100_distractor_3

### 5-way 1-shot 100-unlabel
#### TransMatch 
python main.py --gpu 0 --num-way 5 --num-sample 1 --unlabelnumber 100 --epoch 20 --checkpoint TransMatch_5_1_100

#### MixMatch 
python main.py --gpu 0 --random --num-way 5 --num-sample 1 --unlabelnumber 100 --epoch 20 --checkpoint MixMatch_5_1_100

#### TransMatch with distractor class
python main.py --gpu 0 --num-way 5 --num-sample 1 --unlabelnumber 100 --distractor --distractor_class 3 --epoch 20 --checkpoint TransMatch_5_1_100_distractor_3


