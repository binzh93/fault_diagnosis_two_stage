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
import math


def raw_data_make(path_dir):
    os.chdir(path_dir)
    a = []
    list_dir = os.listdir("Norm")
    if ".DS_Store" in list_dir:
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
        if ".DS_Store" in list_dir: 
            list_dir.remove(".DS_Store")   # bad mac
        for i in range(len(list_dir)):
            path = os.path.join(str(k) + "/0.007", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)

            for j in range(100):
                a.append(val[0][j * 1200: (j + 1) * 1200])

        list_dir = os.listdir(str(k) + "/0.014")
        if ".DS_Store" in list_dir: 
            list_dir.remove(".DS_Store")   # bad mac
        for i in range(len(list_dir)):
            path = os.path.join(str(k) + "/0.014", list_dir[i])
            data = sio.loadmat(path)
            val_name = list_dir[i].split(".")[-2]
            val = data["X" + val_name + "_DE_time"].reshape(1, -1)

            for j in range(100):
                a.append(val[0][j * 1200: (j + 1) * 1200])

        list_dir = os.listdir(str(k) + "/0.021")
        if ".DS_Store" in list_dir: 
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


def split_train_val_test(feature_shuffle, sparse_label_shuffle, split_rate=(0.6, 0.2, 0.2)):
    print('----------split train and test----------')
    train_nums = int(4000*split_rate[0])
    val_nums   = int(4000*split_rate[1])
    test_nums  = int(4000*split_rate[2])
    # train
    train_X = feature_shuffle[: train_nums]
    train_Y = sparse_label_shuffle[: train_nums]
    # val
    val_X = feature_shuffle[train_nums: (train_nums+val_nums)]
    val_Y = sparse_label_shuffle[train_nums: (train_nums+val_nums)]
    # test
    test_X = feature_shuffle[(train_nums+val_nums): ]
    test_Y = sparse_label_shuffle[(train_nums+val_nums): ]

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def generate_npy_file(train_X, train_Y, val_X, val_Y, test_X, test_Y, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, "train_X.npy"), train_X)
    np.save(os.path.join(save_dir, "train_Y.npy"), train_Y)
    np.save(os.path.join(save_dir, "val_X.npy"), val_X)
    np.save(os.path.join(save_dir, "val_Y.npy"), val_Y)
    np.save(os.path.join(save_dir, "test_X.npy"), test_X)
    np.save(os.path.join(save_dir, "test_Y.npy"), test_Y)

def load_npy_file(npy_dir):
    train_X = np.load(os.path.join(npy_dir, "train_X.npy"))
    train_Y = np.load(os.path.join(npy_dir, "train_Y.npy"))
    val_X   = np.load(os.path.join(npy_dir, "val_X.npy"))
    val_Y   = np.load(os.path.join(npy_dir, "val_Y.npy"))
    test_X  = np.load(os.path.join(npy_dir, "test_X.npy"))
    test_Y  = np.load(os.path.join(npy_dir, "test_Y.npy"))
    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def get_fft_feature(train_X, val_X, test_X, isHalf=True, isPostive=True):
    if isPostive:
        fea_train = abs(np.fft.fft(train_X))
        fea_val   = abs(np.fft.fft(val_X))
        fea_test  = abs(np.fft.fft(test_X))
    else:
        fea_train = np.fft.fft(train_X)
        fea_val   = np.fft.fft(val_X)
        fea_test  = np.fft.fft(test_X)
    if isHalf:
        half_len = fea_train[0].size/2
#         half_len = fea_train[0].size
        return fea_train[:, 0: half_len], fea_val[:, 0: half_len], fea_test[:, 0: half_len]
    else:
        return fea_train, fea_val, fea_test



def get_wavelet_packet_decomposition_feature(train_X, val_X, test_X, fs=12000):
    train_num = train_X.shape[0]
    val_num   = val_X.shape[0]
    feature = np.concatenate((train_X, val_X, test_X), axis=0)
    fea = []
    data_size = feature.shape[0]
    for i in range(data_size):
        wave = Wavelet_analyze(feature[i], fs=fs)
        wave.wavelet_tree()
        val = wave.wprvector()
        fea.append(val)
    fea = np.array(fea)
    print(fea.shape)
    return fea[0: train_num], fea[train_num: (train_num+val_num)], fea[(train_num+val_num): ]


def get_single_time_fea(array_fea):
    array_size = array_fea.shape[0]
    array_mean = np.mean(array_fea, axis=1)
    array_std  = np.std(array_fea, axis=1)

    rms = np.zeros((array_size, ))
    K = np.zeros((array_size, ))
    for i in range(array_size):
        for j in range(array_fea.shape[1]):
            rms[i] += array_fea[i][j]*array_fea[i][j]
            K[i]   += math.pow((array_fea[i][j]-array_mean[i]), 4)/ math.pow(array_std[i], 4)
    rms = np.sqrt(rms * 1.0 / array_fea.shape[1])
    K = K * 1.0 / array_fea.shape[1]
    S = rms*1.0/ abs(array_mean)
    C = np.max(array_fea, axis=1) * 1.0 / rms
    I = np.max(array_fea, axis=1)*1.0/abs(array_mean)
    array_mean = array_mean.reshape(array_size, 1)
    array_std  = array_std.reshape(array_size, 1)
    rms        = rms.reshape(array_size, 1)
    K          = K.reshape(array_size, 1)
    S          = S.reshape(array_size, 1)
    C          = C.reshape(array_size, 1)
    I          = I.reshape(array_size, 1)
    time_fea = np.concatenate((array_mean, array_std, rms, K, S, C, I), axis=1)
    return time_fea

def get_time_fea(train_X, val_X, test_X):
    train_X = get_single_time_fea(train_X)
    val_X = get_single_time_fea(val_X)
    test_X = get_single_time_fea(test_X)
    return train_X, val_X, test_X

def fft_wp_fea(train_X, val_X, test_X, isHalf=True, isPostive=True, fs=12000):
    train_fft, val_fft, test_fft = get_fft_feature(train_X, val_X, test_X, isHalf=True, isPostive=True)
    train_wp, val_wp, test_wp    = get_wavelet_packet_decomposition_feature(train_X, val_X, test_X, fs=12000)
    
    train_X = np.concatenate((train_fft, train_wp), axis=1)
    val_X   = np.concatenate((val_fft, val_wp), axis=1)
    test_X  = np.concatenate((test_fft, test_wp), axis=1)

    return train_X, val_X, test_X





if __name__ == "__main__":
    # path_dir = 'castle_data'
    # all_data, label, sparse_label = raw_data_make(path_dir)
    # feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    # train_X, train_Y, test_X, test_Y = split_train_test(feature_shuffle, sparse_label_shuffle, split_rate=0.6)
    
    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)
    # wave_train_X, wave_test_X = get_wavelet_packet_decomposition_feature(train_X, test_X)
    # print(wave_train_X.shape)
    # print(wave_test_X.shape)

    path_dir = 'castle_data'
    # step1 : mat file to npy array 
    all_data, label, sparse_label = raw_data_make(path_dir)
    # step2: get shuffile array and corresponding label 
    feature_shuffle, sparse_label_shuffle = get_shuffle_feature_and_laebl(all_data, sparse_label)
    # step3: split array to train,val,test array and label
    train_X, train_Y, val_X, val_Y, test_X, test_Y = split_train_val_test(feature_shuffle, 
                                                                        sparse_label_shuffle, 
                                                                        split_rate=(0.6, 0.2, 0.2))
    # get_time_fea(train_X, val_X, test_X)
    get_time_fea(train_X, val_X, test_X)
    
    
    
    
    
    
    
    
    
    
    
    # # step4: generate temp npy file
    # save_dir = "../npy_file"
    # generate_npy_file(train_X, train_Y, val_X, val_Y, test_X, test_Y, save_dir)
    # # step5: load npy file
    # npy_dir = save_dir
    # train_X, train_Y, val_X, val_Y, test_X, test_Y = load_npy_file(npy_dir)
    # # fft feature
    # # train_fft_X, val_fft_X, test_fft_X = get_fft_feature(train_X, val_X, test_X, isHalf=True, isPostive=True)
    # # wave packet feature
    # train_wp_X, val_wp_X, test_wp_X = get_wavelet_packet_decomposition_feature(train_X, val_X, test_X, fs=12000)
   



    # print(all_data.shape)
    # print(label.shape)
    # print(sparse_label.shape)
    # print('--------')
    # print(feature_shuffle.shape)
    # print(sparse_label_shuffle.shape)
    # print('--------')
    # print(train_X.shape)
    # print(train_Y.shape)
    # print(val_X.shape)
    # print(val_Y.shape)
    # print(test_X.shape)
    # print(test_Y.shape)
    # print('--------')
    # # print(train_fft_X.shape)
    # # print(val_fft_X.shape)
    # # print(test_fft_X.shape)
    # print('--------')
    # print(train_wp_X.shape)
    # print(val_wp_X.shape)
    # print(test_wp_X.shape)
    # print('--------')





