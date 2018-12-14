# 2018ECE285final
## Description
This is the project Aerial photography image segmentation withVGG16-FCN which developed by team composed of HAO-KUN WU, HSUAN-CHU LIN, SHIH-CHEN LIN, YA-YU LIN. 


## Requirements
- Python:3.6 
- Tensorflow:1.6  
- Keras:2.0.7 
- Scikit-image and Matplotlib standard library  


## Code Organizations
- final.py                -- main programming file  
- dataloader.py           -- tools for loading training and test data of aerial images  
- FCN32.py                -- model construction for VGG-FCN32
- FCN8.py                 -- model construction for VGG-FCN8
- VGGUnet.py              -- model construction for VGG-UNet
- predict.py              -- predicting the testing results
- mean_iou_evaluate.py    -- evaluating the testing results


## Code Usages
1. Training:
```
python3 hw3.py --validate --save_weights_path $1 --train_images $2 --train_annotations $3 --val_images $4 --val_annotations $5 --epochs # --model_name $6
```
$1: the directory where you want to save model weights  
$2: the directory of training images  
$3: the directory of training labels (image masks)  
$4: the directory of validation image  
$5: the directory of validation labels (image masks)  
$6: specify the training model: VGG16-FCN32, VGG16-FCN8, VGG16-UNet  

2. Testing:
```
python3 predict.py --save_weights_path $1 --test_images $2 --output_path $3 --model_name $4
```
$1: the model parameters you want to test on  
$2: the directory of testing images  
$3: the directory of predicting results  
$4: specify the training model  

3. Evaluation:
```
python3 mean_iou_evaluate.py -g $1 -p $2
```
$1: ground truth image masks for aerial images  
$2: predicting image masks  
