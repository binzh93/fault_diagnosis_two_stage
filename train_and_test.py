# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.svm import SVC
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
# from xgboost.sklearn import XGBClassifier
from data_preprocess import *
from sklearn.linear_model import LogisticRegression
from sdae_test import *
from two_stage_sdae import *
import argparse


def svm_test(train_X, train_Y, test_X, test_Y):
    train_Y = np.argmax(train_Y, axis=1)
    test_Y = np.argmax(test_Y, axis=1)

    #clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='rbf', coef0=0.0, probability=True)
    #clf = SVC(kernel='rbf', C=1e3, gamma=0.1)
    clf = SVC(C=1e3)

    #clf.fit(train_X, train_Y, eval_metric='auc')
    clf.fit(train_X, train_Y)

    test_pre = clf.predict(test_X)
    print("SVM accuracy: {}".format(metrics.accuracy_score(test_Y, test_pre)))


def xgboost_test(train_X, train_Y, test_X, test_Y):
    clf = XGBClassifier(
        learning_rate =0.05,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,   # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言, 
                                #假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                                #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        gamma=0,              # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=0.8,          # 随机采样训练样本 训练实例的子采样比
        colsample_bytree=0.8,   # 生成树时进行的列采样 
        objective= 'binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        seed=27)
    train_Y = np.argmax(train_Y, axis=1)
    test_Y = np.argmax(test_Y, axis=1)

    clf.fit(train_X, train_Y)

    test_pre = clf.predict(test_X)
    print("Xgboost accuracy: {}".format(metrics.accuracy_score(test_Y, test_pre)))
    

def algorithom_compare_by_original_singal(data_dir):
    all_data, label, sparse_label = raw_data_make(data_dir)
    # np.save("/Users/bin/fault_diagnosis/st.npy", all_data[0])
    feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    train_X, train_Y, test_X, test_Y = split_train_test(feature_shuffle, sparse_label_shuffle, split_rate=0.6)
    print("SVM train and test")
    svm_test(train_X, train_Y, test_X, test_Y)
    print("Xgboost train and test")
    xgboost_test(train_X, train_Y, test_X, test_Y)
    print("Two stage train and test")
    # sdae = Stacked_Denoising_AutoEncoder(train_X, train_Y,
    #                                 #  validation_data, validation_label,
    #                                 #  test_data, test_label,
    #                                 test_X, test_Y, 
    #                                  10, 100, 0.01, [500, 500, 500], 0.99)
    # sdae.train_new()
    sdae =  Stacked_Denoising_AutoEncoder(train_X, train_Y,
                                            test_X, test_Y, 
                                            inside_epochs=30, 
                                            inside_batch_size=128,
                                            outside_epochs=80, 
                                            outside_train_batch_size=100,
                                            outside_test_batch_size=100,
                                            inside_learning_rate=0.1, 
                                            learning_rate=0.01,
                                            layer_list=[500, 500, 500], 
                                            nclass=10,
                                            moving_decay=0.99)
    sdae.train()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

def algorithom_compare_by_wave_packet_singal(data_dir):
    all_data, label, sparse_label = raw_data_make(data_dir)
    # np.save("/Users/bin/fault_diagnosis/st.npy", all_data[0])
    feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    train_X, train_Y, test_X, test_Y = split_train_test(feature_shuffle, sparse_label_shuffle, split_rate=0.6)

    wave_train_X, wave_test_X = get_wavelet_packet_decomposition_feature(train_X, test_X)

    print("SVM train and test")
    svm_test(wave_train_X, train_Y, wave_test_X, test_Y)
    print("Xgboost train and test")
    xgboost_test(wave_train_X, train_Y, wave_test_X, test_Y)
    print("Two stage train and test")

    sdae =  Stacked_Denoising_AutoEncoder(wave_train_X, train_Y,
                                            wave_test_X, test_Y, 
                                            inside_epochs=30, 
                                            inside_batch_size=128,
                                            outside_epochs=80, 
                                            outside_train_batch_size=100,
                                            outside_test_batch_size=100,
                                            inside_learning_rate=0.1, 
                                            learning_rate=0.01,
                                            layer_list=[500, 500, 500], 
                                            nclass=10,
                                            moving_decay=0.99)
    sdae.train()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


'''
sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
1.n_components:  PCA算法中所要保留的主成分个数n，缺省时默认为None，所有成分被保留。若赋值为int，
比如n_components=1，将把原始数据降到一个维度。赋值为小数，比如n_components = 0.9，将自动选取特征个数n，使得满足所要求的方差百分比。

2.copy: 类型：bool，True或者False，缺省时默认为True。表示是否在运行算法时，将原始训练数据复制一份,
若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；
若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算。

3.whiten:类型：bool，缺省时默认为False。意义：白化，使得每个特征具有零均值、单位方差

from sklearn.decomposition import PCA     #导入PCA

pca = PCA(0.99,True,True)                             #建立pca类，设置参数，保留99%的数据方差
trainDataS = pca.fit_transform(trainData)             #拟合并降维训练数据
testDataS = pca.transform(testData) 

'''


def two_stage_test(data_dir):
    all_data, label, sparse_label = raw_data_make(data_dir)
    # np.save("/Users/bin/fault_diagnosis/st.npy", all_data[0])
    feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    train_X, train_Y, test_X, test_Y = split_train_test(feature_shuffle, sparse_label_shuffle, split_rate=0.6)

    wave_train_X, wave_test_X = get_wavelet_packet_decomposition_feature(train_X, test_X)

    scaler=preprocessing.StandardScaler().fit(wave_train_X)  
    wave_train_X=scaler.transform(wave_train_X)  
    wave_test_X=scaler.transform(wave_test_X)

    pca = PCA(n_components=100, copy=True, whiten=True) 
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)

    two_stage =  Stacked_Denoising_AutoEncoder_Two_Stage(train_X_up=wave_train_X,
                                                        train_X_down=train_X,
                                                        train_Y=train_Y, 
                                                        test_X_up=wave_test_X,
                                                        test_X_down=test_X,
                                                        test_Y=test_Y,
                                                        inside_epochs=30, 
                                                        inside_batch_size=128, 
                                                        outside_epochs=80, 
                                                        outside_train_batch_size=100,
                                                        outside_test_batch_size=100,
                                                        inside_learning_rate=0.1, 
                                                        learning_rate=0.005, 
                                                        layer_list_up=[500, 500, 500, 500], 
                                                        layer_list_down=[500, 500, 500, 500],
                                                        nclass=10, 
                                                        moving_decay=0.99)

    two_stage.train()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="command for training two stage sdae neural network")
    parser.add_argument('--data_dir', type=str, default='/workspace/mnt/group/face-reg/zhubin/fault_diagnosis/fault_data', help='input data dir')
    args = parser.parse_args()

    # algorithom_compare_by_original_singal()
    # algorithom_compare_by_wave_packet_singal()
    two_stage_test(args.data_dir)

    
    
