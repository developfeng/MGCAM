"""
Market-1501 and CUHK-03 training data.

by Chunfeng Song

2017/10/08

This code is for research use only, please cite our paper:
 
Chunfeng Song, Yan Huang, Wanli Ouyang, and Liang Wang. Mask-guided Contrastive Attention Model for Person Re-Identification. In CVPR, 2018.
 
Contact us: chunfeng.song@nlpr.ia.ac.cn
"""

import os
import shutil
import numpy as np

dataset = 'cuhk03-np/detected' #'cuhk03-np/labeled'  or 'cuhk03-np/detected', cuhk03-np can be download from https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP
data_path = './' + dataset +'/bounding_box_train' #Path for original RGB training set.
seg_path = './' + dataset +'/bounding_box_train_seg' #Path for binary masks of training set.
save_data_path = './' + dataset +'/bounding_box_train_fold'
save_seg_path = './' + dataset +'/bounding_box_train_seg_fold'
if not os.path.exists(save_data_path):
    os.mkdir(save_data_path)
    os.mkdir(save_seg_path)
pic_im_list = np.sort(os.listdir(data_path))
i = 0
for pic in pic_im_list:
    if pic.lower().endswith('.jpg') or pic.lower().endswith('.png'):
        this_data_fold = os.path.join(save_data_path,pic[:4])
        this_seg_fold = os.path.join(save_seg_path,pic[:4])
        if not os.path.exists(this_data_fold):
            os.mkdir(this_data_fold)
            os.mkdir(this_seg_fold)
            i +=1
        new_im_path = os.path.join(this_data_fold,pic)
        new_seg_path = os.path.join(this_data_fold,pic[:-4] + '.png')
        shutil.copy(os.path.join(data_path,pic),new_im_path)
        shutil.copy(os.path.join(seg_path,pic[:-4] + '.png'),this_seg_fold)
        print '---->dealing num-%04d with %s!'%(i,pic)
print 'DONE!!!'

