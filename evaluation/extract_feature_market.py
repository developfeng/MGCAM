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
    prefix_list = ['query', 'bounding_box_test','bounding_box_train']
    prefix_2 = None #None or 'siamese'
    gpu_id = 0
    if prefix_2 is None:
        model_data = '../experiments/market1501/mgcam_iter_75000.caffemodel'
    else:
        model_data = '../experiments/market1501/mgcam_'+prefix_2+'_iter_20000.caffemodel'
    fea_dims = 128
    model_config = './deploy_mgcam.prototxt'
    data_path = '../data/market1501/'
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(model_config, model_data, caffe.TEST)
    for prefix in prefix_list:
        image_path = os.path.join(data_path, prefix)
        mask_path = os.path.join(data_path, prefix + '_seg')
        # list images
        image_list = n.sort(os.listdir(image_path))
        length = len(image_list)-1
        feature_all = n.zeros((fea_dims,length),n.single)
        now = 0
        for item in image_list:
            if item.lower().endswith('.jpg'):
                this_image_path = os.path.join(image_path,item)
                this_mask_path = os.path.join(mask_path,item[:-4] + '.png')
                this_fea = extract_feature(net,this_image_path,this_mask_path)
                feature_all[:,now] = this_fea[:]
                now +=1
                print '---->%04d of %d  with %s is done!'%(now,len(image_list),item)
        if prefix_2 is None:
            spio.savemat('market-fea-' + prefix, {'feat': feature_all})
        else:
            spio.savemat('market-fea-' + prefix +'-'+ prefix_2, {'feat': feature_all})
