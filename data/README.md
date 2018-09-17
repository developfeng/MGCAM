Dataset Preparation.

1) Download MARS dataset from [http://www.liangzheng.com.cn/Project/project_mars.html].

2) Download Market-1501 dataset from [http://www.liangzheng.org/Project/project_reid.html].

3) Download CUHK03 dataset from [https://github.com/zhunzhong07/person-re-ranking]. You need to extract the images into folods. You can also download the new protocol version (CUHK03-NP) from [https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP].

* All masks can be download from Baidu Yun (https://pan.baidu.com/s/16ZrlM1f_1_T-eZHmQTTkYg) OR Google Drive(https://drive.google.com/drive/folders/1QVBDpH0B4k6cXKFYXBJ3HNVET_3gY0to?usp=sharing).

Make sure that all the datasets are saveing in the following structure:

MARS:
./data
./data/mars
./data/mars/bbox_train
./data/mars/bbox_test
./data/mars/bbox_train_seg
./data/mars/bbox_test_seg

Market-1501:
./data
./data/market-1501
./data/market-1501/bounding_box_train
./data/market-1501/bounding_box_test
./data/market-1501/query
./data/market-1501/bounding_box_train_seg
./data/market-1501/bounding_box_test_seg
./data/market-1501/query_seg

CUHK03:
./data
./data/cuhk03
./data/cuhk03/labeled
./data/cuhk03/cuhk03_labeled_seg
./data/cuhk03/detected
./data/cuhk03/cuhk03_detected_seg

CUHK03-NP:
./data
./data/cuhk03-np
./data/cuhk03-np/labeled
./data/cuhk03-np/labeled/bounding_box_train
./data/cuhk03-np/labeled/bounding_box_test
./data/cuhk03-np/labeled/query
./data/cuhk03-np/labeled/bounding_box_train_seg
./data/cuhk03-np/labeled/bounding_box_test_seg
./data/cuhk03-np/labeled/query_seg

./data/cuhk03-np/detected
./data/cuhk03-np/detected/bounding_box_train
./data/cuhk03-np/detected/bounding_box_test
./data/cuhk03-np/detected/query
./data/cuhk03-np/detected/bounding_box_train_seg
./data/cuhk03-np/detected/bounding_box_test_seg
./data/cuhk03-np/detected/query_seg

Now, you can run the code to generate training set with running "python gen_market1501_and_cuhk03_train_set.py"