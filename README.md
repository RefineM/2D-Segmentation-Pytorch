<h1>Segment_Pytorch</h1>

<h4> Simply implement image semantic segmentation based on pytorch.</h4>

## Code Structure
```
--data_process
  |-- ori_data         
  |   |-- img
  |   |-- mask
  |-- dataset            
  |   |-- train
  |   |   |-- img
  |   |   |-- mask  
  |   |-- val
  |   |   |-- img
  |   |   |-- mask
  |   |-- test
  |   |   |-- img
  |   |   |-- mask
--model                   
  |-- block.py
  |-- encoder.py
  |-- seg_model.py
--utils
  |-- dataset_load.py    
  |-- dataset_spilt.py   
  |-- loss.py            
  |-- metric.py           
  |-- optimizer.py
  |-- labelRGB.py       
--opt.py
--train.py
--predict.py

```

* Net：
   * Unet
* Loss function：
   * Cross-Entropy loss 
   * Dice loss
   * Focal loss
* Metric：
   * IOU
   * Dice-Score
   * Accuracy
   * Confusion Matrix

## Environment Configuration
1. Install Anaconda
2. Install CUDA
3. Create and activate virtual environment  
   ```
     conda create -n [name] python==3.8
     conda activate [name]
   ```
4. Install torch_gpu (cuda 11.3)
   ```
     pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
   ```
5. Install other packages
   ```
     pip install wandb  
     pip install pillow
     pip install tqdm
     pip install numpy
   ```
   
## Dataset
The structure of dataset is:
```
  |-- dataset            
  |   |-- train
  |   |   |-- img
  |   |   |-- mask  
  |   |-- val
  |   |   |-- img
  |   |   |-- mask
  |   |-- test
  |   |   |-- img
  |   |   |-- mask
```
Put the images under `data_process/ori_data/img` and put masks under `data_process/ori_data/mask`, then change the arguments in `opt.py`:
* Randomly devide the original dataset into train/val/test dataset in centain proportion:
    * Firstly set `--if_split` to `True`
    * Then set the scale of train/val/test dataset by changing the value of `--train_scale` and `--val_scale` (the values range from 0 to 1).
      if `--train_scale` is 0.8 and `--val_scale` is 0.1, the scale of test set is automatically set to 0.1
* (Optional) Trim the imgs and masks of train and val dataset into a specified size:
    * Firstly set `--if_crop` to `True`
    * Set the target size of images by changing the value of `--target_size`. For example, 256
    * Set the amount of small images cropped from each large image by changing the value of `--target_num`. For example, 8
* (Optional) Apply data enhancement to train dataset:
    * Set `--if_enhance` to `True`
    * Set the scale of images that apply enhancement by changing `--enhance_scale` (the value ranges from 0 to 1)
      
If you need to change the methods of implementing the above functions, please modify the class `CropSplitTool` in `dataset_spilt.py`.

## Train
1. Change the arguments in `opt.py`
   * To use gpu, change `--device` to ' cuda:id '
   * To use pre-trained weights, change `--if_pre_ckpt` to `True` , and set `--pre_ckpt_path`
2. Run `train.py`
3. The `checkpoint_epoch_x.pth` of each epoch will be put under the folder `checkpoint`

## Predict
1. Change the arguments in `opt.py`
   * Set the path of weight file `--ckpt_path`
   * To save predicted results, please set `--if_save` to `True`, and modify the path `--output_dir` 
   * To output RGB results, please modify color maps in `labelRGB.py` and set `--if_labelRGB` to `True`  
2. run `predict.py`
   
## Test
1. ***WHU Building Dataset (Satellite dataset I)***  [Download](http://gpcv.whu.edu.cn/data/building_dataset.html)

   |size|classnum|train|val|test|
   |:--:|:--:|:--:|:--:|:--:|   
   |512*512|2|202|66|67|  

   |no.|net|batchsize|epoch|lr|optimizer|lr-scheduler|loss|IOU(%)|Dice(%)|Acc(%)|  
   |:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|  
   |1|UNet|8|100|1e-3|AdamW|Cosine|CE-loss|96.09|97.99|98.58|

   [Result (Google drive)](https://drive.google.com/drive/folders/1LNUMpvLCm_GKu17c7Mn4Dvl9S5fp0JvK?usp=drive_link)
  
## Reference
* u-net：
  https://github.com/milesial/Pytorch-UNet
* LoveDA：
  https://github.com/Junjue-Wang/LoveDA
* loss func and metric：
  https://github.com/open-mmlab/mmsegmentation
