"""
Extracting features with caffe models.

by Chunfeng Song

2017/10/08

This code is for research use only, please cite our paper:
 
Chunfeng Song, Yan Huang, Wanli Ouyang, and Liang Wang. Mask-guided Contrastive Attention Model for Person Re-Identification. In CVPR, 2018.
 
Contact us: chunfeng.song@nlpr.ia.ac.cn
"""
import caffe
import numpy as n
import cv2
import scipy.io as spio
import os

def extract_feature(net,image_path, gt_path):
    # load image
    oim = cv2.imread(image_path)

    # resize image into caffe size
    inputImage = cv2.resize(oim, (64, 160))    
    inputImage = n.array(inputImage, dtype=n.float32)
    
    # substract mean
    inputImage[:, :, 0] = inputImage[:, :, 0] - 104.008
    inputImage[:, :, 1] = inputImage[:, :, 1] - 116.669
    inputImage[:, :, 2] = inputImage[:, :, 2] - 122.675
    
    # permute dimensions
    inputImage = inputImage.transpose([2, 0, 1])
    inputImage = inputImage/256.0
    one_mask_ = n.zeros((1,40,16),dtype = n.float32)
    one_mask_ = one_mask_ + 1.0
    
    # mask
    mask_im= cv2.cvtColor(cv2.imread(gt_path),cv2.COLOR_BGR2GRAY)
    inputmask = n.array(cv2.resize(mask_im, (64, 160)), dtype=n.float32)
    inputmask = inputmask-127.5
    inputmask = inputmask/255.0
    inputmask = inputmask[n.newaxis, ...]
    
    #caffe forward 
    net.blobs['data'].reshape(1,*inputImage.shape)
    net.blobs['one_mask'].reshape(1,*one_mask_.shape)
    net.blobs['mask'].reshape(1,*inputmask.shape)
    net.blobs['data'].data[...] = inputImage
    net.blobs['one_mask'].data[...] = one_mask_
    net.blobs['mask'].data[...] = inputmask
    net.forward()
    
    #caffe output
    feature = n.squeeze(net.blobs['fc1_full'].data)
    return feature

if __name__ == '__main__':
    pass
    prefix = 'labeled' #'labeled' or 'detected'
    prefix_2 = None #None or 'siamese'
    gpu_id = 0
    if prefix_2 is None:
        model_data = '../experiments/cuhk03-'+prefix+'/mgcam_iter_75000.caffemodel'
    else:
        model_data = '../experiments/cuhk03-'+prefix+'/mgcam_'+prefix_2+'_iter_20000.caffemodel'
    fea_dims = 128
    model_config = './deploy_mgcam.prototxt'
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(model_config, model_data, caffe.TEST)
    image_path = '../data/cuhk03/cuhk03_'+prefix
    mask_path = '../data/cuhk03/cuhk03_'+ prefix + '_seg'
    # list images
    image_list = n.sort(os.listdir(image_path))
    feature_all = n.zeros((fea_dims,len(image_list)),n.single)
    now = 0
    for item in image_list:
        if item.lower().endswith('.png'):
            this_image_path = os.path.join(image_path,item)
            this_mask_path = os.path.join(mask_path,item)
            this_fea = extract_feature(net,this_image_path,this_mask_path)
            feature_all[:,now] = this_fea[:]
            now +=1
            print '---->%04d of %d  with %s is done!'%(now,len(image_list),item)
    if prefix_2 is None:
        spio.savemat('cuhk03-fea-' + prefix, {'feat': feature_all})
    else:
        spio.savemat('cuhk03-fea-' + prefix + '-' + prefix_2, {'feat': feature_all})
