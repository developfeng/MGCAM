# MGCAM
--------------------------------------------------------------------------------
* Mask-guided Contrastive Attention Model (MGCAM) for Person Re-Identification 
* Code Version 1.0                                                                                                                       
* E-mail: chunfeng.song@nlpr.ia.ac.cn                                          
---------------------------------------------------------------------------------

i.    Overview
ii.   Copying
iii.  Use

i. OVERVIEW
-----------------------------
This code implements the paper:

>Chunfeng Song, Yan Huang, Wanli Ouyang, and Liang Wang. Mask-guided 
Contrastive Attention Model for Person Re-Identification. 
In CVPR, 2018.

If you find this work is helpful for your research, please cite our paper [[PDF]](http://openaccess.thecvf.com/content_cvpr_2018/html/Song_Mask-Guided_Contrastive_Attention_CVPR_2018_paper.html).

ii. COPYING
-----------------------------
We share this code only for research use. We neither warrant 
correctness nor take any responsibility for the consequences of 
using this code. If you find any problem or inappropriate content
in this code, feel free to contact us (chunfeng.song@nlpr.ia.ac.cn).

iii. USE
-----------------------------
This code should work on Caffe with Python layer (pycaffe). You can install Caffe from [here](https://github.com/BVLC/caffe).

(1) Data Preparation.   
Download the original datasets and their masks: MARS, Market-1501, CUHK-03, and their masks from [Baidu Yun](https://pan.baidu.com/s/16ZrlM1f_1_T-eZHmQTTkYg) OR [Google Drive](https://drive.google.com/drive/folders/1QVBDpH0B4k6cXKFYXBJ3HNVET_3gY0to?usp=sharing).

For Market-1501 and CUHK-03, you need to run the spilt code(./data/gen_market1501_and_cuhk03_train_set.py).

(2) Model Training. 
Here, we take MARS as an example. The other two datasets are the same.

>cd ./experiments/mars

First eidt the 'im_path', 'gt_path' and 'dataset' in the prototxt file, e.g., the MGCAM-only and MGCAM-Siamese version for MARS dataset is 'mgcam_train.prototxt' and 'mgcam_siamese_train.prototxt', respectively. 

Then, we can train the MGCAM model from scratch with the command:
>sh run_mgcam.sh  

It will take roughly 15 hours for single Titan X.

Finally, we can fine-tune the MGCAM model with siamese loss via run the commman:
>sh run_mgcam_siamese.sh 

It will take roughly 5 hours for single Titan X.

(3) Evaluation.   
Taking MARS for example, run the code in './evaluation/extract_feature_mars.py' to extract the IDE features, and then run the CMC and mAP evaluation with the [MARS-evaluation](https://github.com/liangzheng06/MARS-evaluation) code by Liang Zheng et al., or the [Re-Ranking](https://github.com/zhunzhong07/person-re-ranking) by Zhun Zhong et al.
