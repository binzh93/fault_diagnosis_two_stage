# -*- coding: utf-8 -*-
'''
split the raw data to 1200 sample points
Noraml: 400*1200   400(sample nums)  1200(sample points)
Ball: 3*400*1200   3(falut category) 400(sample nums)  1200(sample points)
Inner Race: 3*400*1200   3(falut category) 400(sample nums)  1200(sample points)
Outer Race: 3*400*1200   3(falut category) 400(sample nums)  1200(sample points)
'''
import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import os
import time
from Wavelet import *


#os.chdir("/Users/bin/Desktop/fault diagnosis/毕业论文相关/未命名文件夹/fault diagnosis/fault_data")
# os.chdir("/workspace/mnt/group/face-reg/zhubin/fault_diagnosis/fault_data")


def raw_data_make(path_dir):
    os.chdir(path_dir)
    a = []
    list_dir = os.listdir("Norm")
    list_dir.remove(".DS_Store")    # bad mac
    # get Noraml feature

    for i in range(len(list_dir)):
        path = os.path.join("Norm", list_dir[i])
        data = sio.loadmat(path)
        val_name = list_dir[i].split(".")[-2]
        if len(val_name)==2:
            val = data['X0' + val_name + "_DE_time"].reshape(1, -1)
        else:
            val = data['X' + val_name + "_DE_time"].reshape(1, -1)
        for j in range(100):
            a.append(val[0][j * 1200: (j + 1) * 1200])

    # get fault feature
    cat_list = ["Ball", "Inner Race", "Outer Race"]
    for k in cat_list:
        list_dir = os.listdir(str(k) + "/0.007")
        list_dir.remove(".DS_Store")   # bad mac
        for i in range(len(list_dir)):
            path = os.path.join(str(k) + "/0.007", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)

            for j in range(100):
                a.append(val[0][j * 1200: (j + 1) * 1200])

        list_dir = os.listdir(str(k) + "/0.014")
        list_dir.remove(".DS_Store")   # bad mac
        for i in range(len(list_dir)):
            path = os.path.join(str(k) + "/0.014", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)

            for j in range(100):
                a.append(val[0][j * 1200: (j + 1) * 1200])

        list_dir = os.listdir(str(k) + "/0.021")
        list_dir.remove(".DS_Store")   # bad mac
        for i in range(len(list_dir)):
            path = os.path.join(str(k) + "/0.021", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)

            for j in range(100):
                a.append(val[0][j * 1200: (j + 1) * 1200])
    all_data = np.array(a)  # 全部特征

    # get single value label
    label = []
    for i in range(10):
        tmp = [i for j in range(400)]
        label.extend(tmp)
    label = np.array(label).reshape(-1, 1)  # 标签 0 1 2 3 4 5 6 7 8 9

    # get one hot label
    enc = preprocessing.OneHotEncoder()
    enc.fit(label)
    arr = enc.transform(label).toarray()
    sparse_label = np.array(arr, dtype="int64")  # 稀疏标签

    return all_data, label, sparse_label


def get_shuffle_feature_and_laebl(feature, label):
    data = np.concatenate((feature, label), axis=1)
    np.random.shuffle(data)
    feature = data[:, 0: 1200]
    label = data[:, 1200:]
    return feature, label


def get_train_validation_test(all_data, sparse_label):
    for i in range(10):
        if i == 0:
            train_tmp = all_data[0: 280]
            validation_tmp = all_data[280: 360]
            test_tmp = all_data[360: 400]

            train_label_tmp = sparse_label[0: 280]
            validation_label_tmp = sparse_label[280: 360]
            test_label_tmp = sparse_label[360: 400]
        else:
            train_tmp = np.concatenate((train_tmp, all_data[i * 400: i * 400 + 280]), axis=0)
            validation_tmp = np.concatenate((validation_tmp, all_data[i * 400 + 280: i * 400 + 360]), axis=0)
            test_tmp = np.concatenate((test_tmp, all_data[i * 400 + 360: (i + 1) * 400]), axis=0)

            train_label_tmp = np.concatenate((train_label_tmp, sparse_label[i * 400: i * 400 + 280]), axis=0)
            validation_label_tmp = np.concatenate((validation_label_tmp, sparse_label[i * 400 + 280: i * 400 + 360]),
                                                  axis=0)
            test_label_tmp = np.concatenate((test_label_tmp, sparse_label[i * 400 + 360: (i + 1) * 400]), axis=0)

    train = np.array(train_tmp)
    validation = np.array(validation_tmp)
    test = np.array(test_tmp)

    train_label = np.array(train_label_tmp)
    validation_label = np.array(validation_label_tmp)
    test_label = np.array(test_label_tmp)
    return train, train_label, test, test_label
    

def split_train_test(feature_shuffle, sparse_label_shuffle, split_rate=0.6):
    print('----------split train and test----------')
    # split_rate = 0.6
    split_nums = int(4000*split_rate)
    train_X = feature_shuffle[: split_nums]
    train_Y = sparse_label_shuffle[: split_nums]
    test_X = feature_shuffle[split_nums:]
    test_Y = sparse_label_shuffle[split_nums:]
    return train_X, train_Y, test_X, test_Y


def get_original_feature_after_pca(train_X, test_X):
    pass


def get_wavelet_packet_decomposition_feature(train_X, test_X):
    split_num = train_X.shape[0]
    feature = np.concatenate((train_X, test_X), axis=0)
    fea = []
    for i in range(4000):
        wave = Wavelet_analyze(feature[i], fs=12000)
        wave.wavelet_tree()
        val = wave.wprvector()
        fea.append(val)
    fea = np.array(fea)
    return fea[0: split_num], fea[split_num: ]

def get_fft_feature(train_X, test_X):
    fea_train = np.fft.fft(train_X)
    fea_test = np.fft.fft(test_X)
    return fea_train, fea_test
    

if __name__ == "__main__":
    all_data, label, sparse_label = raw_data_make()
    feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    train_X, train_Y, test_X, test_Y = split_train_test(feature_shuffle, sparse_label_shuffle, split_rate=0.6)
    
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    wave_train_X, wave_test_X = get_wavelet_packet_decomposition_feature(train_X, test_X)
    print(wave_train_X.shape)
    print(wave_test_X.shape)

