"""
MGCAM data layer.

by Chunfeng Song

2017/10/08

This code is for research use only, please cite our paper:
 
Chunfeng Song, Yan Huang, Wanli Ouyang, and Liang Wang. Mask-guided Contrastive Attention Model for Person Re-Identification. In CVPR, 2018.
 
Contact us: chunfeng.song@nlpr.ia.ac.cn
"""

import caffe
import numpy as np
import yaml
from random import shuffle
import numpy.random as nr
import cv2
import os
import pickle as cPickle
import pdb

def mypickle(filename, data):
    fo = open(filename, "wb")
    cPickle.dump(data, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()
    
def myunpickle(filename):
    if not os.path.exists(filename):
        raise UnpickleError("Path '%s' does not exist." % filename)

    fo = open(filename, 'rb')
    dict = cPickle.load(fo)    
    fo.close()
    return dict

class MGCAM_DataLayer(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top): 
        self.width = 64
        self.height = 160 # We resize all images into a size of 160*64.
        self.width_gt = 16
        self.height_gt = 40 # We resize all masks which are used to supervise attention learning into a size of 40*16.
        layer_params = yaml.load(self.param_str)
        self.batch_size = layer_params['batch_size']
        self.im_path = layer_params['im_path']
        self.gt_path = layer_params['gt_path']
        self.dataset = layer_params['dataset']
        self.labels, self.im_list, self.gt_list, self.im_dir_list, self.gt_dir_list = self.data_processor(self.dataset)
        self.idx = 0
        self.data_num = len(self.im_list) # Number of data pairs
        self.rnd_list = np.arange(self.data_num) # Random the images list
        shuffle(self.rnd_list)
        
    def forward(self, bottom, top):
        # Assign forward data
        top[0].data[...] = self.im
        top[1].data[...] = self.inner_label
        top[2].data[...] = self.exter_label
        top[3].data[...] = self.one_mask
        top[4].data[...] = self.label
        top[5].data[...] = self.label_plus
        top[6].data[...] = self.gt
        top[7].data[...] = self.mask

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # Load image + label image pairs
        self.im = []
        self.label = []
        self.inner_label = []
        self.exter_label = []
        self.one_mask = []
        self.label_plus = []
        self.gt = []
        self.mask = []

        for i in xrange(self.batch_size):
            if self.idx == self.data_num:
                self.idx = 0
                shuffle(self.rnd_list) #Randomly shuffle the list.
            cur_idx = self.rnd_list[self.idx]
            im_path = self.im_list[cur_idx]
            gt_path = self.gt_list[cur_idx]
            im_, gt_, mask_= self.load_data(im_path, gt_path)
            self.im.append(im_)
            self.gt.append(gt_)
            self.mask.append(mask_)
            self.label.append(self.labels[cur_idx])
            self.inner_label.append(int(1))
            self.exter_label.append(int(0))
            one_mask_ = np.zeros((1,40,16),dtype = np.float32)
            one_mask_ = one_mask_ + 1.0
            self.one_mask.append(one_mask_)
            self.label_plus.append(self.labels[cur_idx]) #Here, we also give the ID-labels to background-stream.
            self.idx +=1

        self.im = np.array(self.im).astype(np.float32)
        self.inner_label = np.array(self.inner_label).astype(np.float32)
        self.exter_label = np.array(self.exter_label).astype(np.float32)
        self.one_mask = np.array(self.one_mask).astype(np.float32)
        self.label = np.array(self.label).astype(np.float32)
        self.label_plus = np.array(self.label_plus).astype(np.float32)
        self.gt = np.array(self.gt).astype(np.float32)
        self.mask = np.array(self.mask).astype(np.float32)
        # Reshape tops to fit blobs      
        top[0].reshape(*self.im.shape)
        top[1].reshape(*self.inner_label.shape)
        top[2].reshape(*self.exter_label.shape)
        top[3].reshape(*self.one_mask.shape)
        top[4].reshape(*self.label.shape)
        top[5].reshape(*self.label_plus.shape)
        top[6].reshape(*self.gt.shape)
        top[7].reshape(*self.mask.shape)
        
    def data_processor(self, data_name):
        data_dic = './' + data_name
        if not os.path.exists(data_dic):
            im_list = []
            gt_list = []
            labels = []
            im_dir_list = []
            gt_dir_list = []
            new_id = 0
            id_list = np.sort(os.listdir(self.im_path))
            for id in id_list:
                im_dir = os.path.join(self.im_path, id)
                gt_dir = os.path.join(self.gt_path, id)
                if not os.path.exists(im_dir):
                    continue
                pic_im_list = np.sort(os.listdir(im_dir))
                if len(pic_im_list)>1:
                    for pic in pic_im_list:
                        this_dir = os.path.join(self.im_path, id, pic)
                        gt_pic = pic
                        if not pic.lower().endswith('.png'):
                            gt_pic = pic[:-4] + '.png'
                        this_gt_dir = os.path.join(self.gt_path, id, gt_pic)
                        im_list.append(this_dir)
                        gt_list.append(this_gt_dir)
                        labels.append(int(new_id))
                    new_id +=1
                    im_dir_list.append(im_dir)
                    gt_dir_list.append(gt_dir)
            dic = {'im_list':im_list,'gt_list':gt_list,'labels':labels,'im_dir_list':im_dir_list,'gt_dir_list':gt_dir_list}
            mypickle(data_dic, dic)
        # Load saved data dict to resume.
        else:
            dic = myunpickle(data_dic)
            im_list = dic['im_list']
            gt_list = dic['gt_list']
            labels = dic['labels']
            im_dir_list = dic['im_dir_list']
            gt_dir_list = dic['gt_dir_list']
        return labels, im_list, gt_list, im_dir_list, gt_dir_list
    
    def load_data(self, im_path, gt_path):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        oim = cv2.imread(im_path)
        inputImage = cv2.resize(oim, (self.width, self.height))
        inputImage = np.array(inputImage, dtype=np.float32)
        
        # Substract mean
        inputImage[:, :, 0] = inputImage[:, :, 0] - 104.008
        inputImage[:, :, 1] = inputImage[:, :, 1] - 116.669
        inputImage[:, :, 2] = inputImage[:, :, 2] - 122.675
        
        # Permute dimensions
        if_flip = nr.randint(2)
        if if_flip == 0: # Also flip the image with 50% probability
            inputImage = inputImage[:,::-1,:]
        inputImage = inputImage.transpose([2, 0, 1])        
        inputImage = inputImage/256.0
        #GT
        mask_im= cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)
        inputGt = np.array(cv2.resize(mask_im, (self.width_gt, self.height_gt)), dtype=np.float32)
        inputGt = inputGt/255.0
        if if_flip == 0:
            inputGt = inputGt[:,::-1]
        inputGt = inputGt[np.newaxis, ...]
        #Mask
        inputMask = np.array(cv2.resize(mask_im, (self.width, self.height)), dtype=np.float32)
        inputMask = inputMask-127.5
        inputMask = inputMask/255.0
        if if_flip == 0:
            inputMask = inputMask[:,::-1]
        inputMask = inputMask[np.newaxis, ...]
        return inputImage, inputGt, inputMask


class MGCAM_SIA_DataLayer(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top): 
        self.width = 64
        self.height = 160 # We resize all images into a size of 160*64.
        self.width_gt = 16
        self.height_gt = 40 # We resize all masks which are used to supervise attention learning into a size of 160*64.
        
        layer_params = yaml.load(self.param_str)
        self.batch_size = layer_params['batch_size']
        self.pos_pair_num = int(0.30*self.batch_size) # There will be at least 30 percent postive pairs for each batch.
        self.im_path = layer_params['im_path']
        self.gt_path = layer_params['gt_path']
        self.dataset = layer_params['dataset']
        self.labels, self.im_list, self.gt_list, self.im_dir_list, self.gt_dir_list = self.data_processor(self.dataset)
        self.idx = 0
        self.data_num = len(self.im_list)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)
        
    def forward(self, bottom, top):
        # Assign forward data
        top[0].data[...] = self.im
        top[1].data[...] = self.inner_label
        top[2].data[...] = self.exter_label
        top[3].data[...] = self.one_mask
        top[4].data[...] = self.label
        top[5].data[...] = self.label_plus
        top[6].data[...] = self.gt
        top[7].data[...] = self.mask
        top[8].data[...] = self.siam_label
        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # Load image + label image pairs
        self.im = []
        self.label = []
        self.inner_label = []
        self.exter_label = []
        self.one_mask = []
        self.label_plus = []
        self.gt = []
        self.mask = []
        self.siam_label = []

        for i in xrange(self.batch_size):
            if self.idx == self.data_num:
                self.idx = 0
                shuffle(self.rnd_list)
            cur_idx = self.rnd_list[self.idx]
            im_path = self.im_list[cur_idx]
            gt_path = self.gt_list[cur_idx]
            im_, gt_, mask_= self.load_data(im_path, gt_path)
            self.im.append(im_)
            self.gt.append(gt_)
            self.mask.append(mask_)
            self.label.append(self.labels[cur_idx])
            self.inner_label.append(int(1))
            self.exter_label.append(int(0))
            one_mask_ = np.zeros((1,40,16),dtype = np.float32)
            one_mask_ = one_mask_ + 1.0
            self.one_mask.append(one_mask_)
            self.label_plus.append(self.labels[cur_idx])#Labels for backgrounds. We use the same labels with other two regions here. 
            self.idx +=1
            
        for i in xrange(self.batch_size):
            if i > self.pos_pair_num:
                if self.idx == self.data_num:
                    self.idx = 0
                    shuffle(self.rnd_list)#Randomly shuffle the list.
                cur_idx = self.rnd_list[self.idx]
                self.idx +=1
                im_path = self.im_list[cur_idx]
                gt_path = self.gt_list[cur_idx]
                label = self.labels[cur_idx]
                if label==self.label[i]:
                    self.siam_label.append(int(1))#In case of getting postive pairs, maybe not much.
                else:
                    self.siam_label.append(int(0))#Negative pairs.
            else:
                im_dir = self.im_dir_list[self.label[i]]
                gt_dir = self.gt_dir_list[self.label[i]]
                im_list = np.sort(os.listdir(im_dir))
                gt_list = np.sort(os.listdir(gt_dir))
                tmp_list = np.arange(len(im_list))
                shuffle(tmp_list) #Randomly select one.
                im_path = os.path.join(im_dir, im_list[tmp_list[0]])
                gt_path = os.path.join(gt_dir, gt_list[tmp_list[0]])
                label = self.label[i]
                self.siam_label.append(int(1))#This is a postive pair.
            
            im_, gt_, mask_= self.load_data(im_path, gt_path)
            self.im.append(im_)
            self.gt.append(gt_)
            self.mask.append(mask_)
            self.label.append(label)
            self.inner_label.append(int(1))#Allways be ones, for constrastive learning.
            self.exter_label.append(int(0))
            one_mask_ = np.zeros((1,40,16),dtype = np.float32)
            one_mask_ = one_mask_ + 1.0
            self.one_mask.append(one_mask_)
            self.label_plus.append(label)
                
        self.im = np.array(self.im).astype(np.float32)
        self.inner_label = np.array(self.inner_label).astype(np.float32)
        self.exter_label = np.array(self.exter_label).astype(np.float32)
        self.one_mask = np.array(self.one_mask).astype(np.float32)
        self.label = np.array(self.label).astype(np.float32)
        self.label_plus = np.array(self.label_plus).astype(np.float32)
        self.gt = np.array(self.gt).astype(np.float32)
        self.mask = np.array(self.mask).astype(np.float32)
        self.siam_label = np.array(self.siam_label).astype(np.float32)
        # Reshape tops to fit blobs      
        top[0].reshape(*self.im.shape)
        top[1].reshape(*self.inner_label.shape)
        top[2].reshape(*self.exter_label.shape)
        top[3].reshape(*self.one_mask.shape)
        top[4].reshape(*self.label.shape)
        top[5].reshape(*self.label_plus.shape)
        top[6].reshape(*self.gt.shape)
        top[7].reshape(*self.mask.shape)
        top[8].reshape(*self.siam_label.shape)
        
    def data_processor(self, data_name):
        data_dic = './' + data_name
        if not os.path.exists(data_dic):
            im_list = []
            gt_list = []
            labels = []
            im_dir_list = []
            gt_dir_list = []
            new_id = 0
            id_list = np.sort(os.listdir(self.im_path))
            for id in id_list:
                im_dir = os.path.join(self.im_path, id)
                gt_dir = os.path.join(self.gt_path, id)
                if not os.path.exists(im_dir):
                    continue
                pic_im_list = np.sort(os.listdir(im_dir))
                if len(pic_im_list)>1:
                    for pic in pic_im_list:
                        this_dir = os.path.join(self.im_path, id, pic)
                        gt_pic = pic
                        if not pic.lower().endswith('.png'):
                            gt_pic = pic[:-4] + '.png'
                        this_gt_dir = os.path.join(self.gt_path, id, gt_pic)
                        im_list.append(this_dir)
                        gt_list.append(this_gt_dir)
                        labels.append(int(new_id))
                    new_id +=1
                    im_dir_list.append(im_dir)
                    gt_dir_list.append(gt_dir)
            dic = {'im_list':im_list,'gt_list':gt_list,'labels':labels,'im_dir_list':im_dir_list,'gt_dir_list':gt_dir_list}
            mypickle(data_dic, dic)
        # Load saved data dict to resume.
        else:
            dic = myunpickle(data_dic)
            im_list = dic['im_list']
            gt_list = dic['gt_list']
            labels = dic['labels']
            im_dir_list = dic['im_dir_list']
            gt_dir_list = dic['gt_dir_list']
        return labels, im_list, gt_list, im_dir_list, gt_dir_list
    
    def load_data(self, im_path, gt_path):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        oim = cv2.imread(im_path)
        inputImage = cv2.resize(oim, (self.width, self.height))
        inputImage = np.array(inputImage, dtype=np.float32)
        
        # Substract mean
        inputImage[:, :, 0] = inputImage[:, :, 0] - 104.008
        inputImage[:, :, 1] = inputImage[:, :, 1] - 116.669
        inputImage[:, :, 2] = inputImage[:, :, 2] - 122.675
        
        # Permute dimensions
        if_flip = nr.randint(2)
        if if_flip == 0: # Also flip the image with 50% probability
            inputImage = inputImage[:,::-1,:]
        inputImage = inputImage.transpose([2, 0, 1])        
        inputImage = inputImage/256.0
        #GT
        mask_im= cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)
        inputGt = np.array(cv2.resize(mask_im, (self.width_gt, self.height_gt)), dtype=np.float32)
        inputGt = inputGt/255.0
        if if_flip == 0:
            inputGt = inputGt[:,::-1]
        inputGt = inputGt[np.newaxis, ...]
        #Mask
        inputMask = np.array(cv2.resize(mask_im, (self.width, self.height)), dtype=np.float32)
        inputMask = inputMask-127.5
        inputMask = inputMask/255.0
        if if_flip == 0:
            inputMask = inputMask[:,::-1]
        inputMask = inputMask[np.newaxis, ...]
        return inputImage, inputGt, inputMask
            