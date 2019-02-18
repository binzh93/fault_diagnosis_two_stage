# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
# import scipy.io as sio
from Wavelet import *
import math




def csv_to_npy_file(npy_save_path):
    #  cause of slow 
    file1 = "jiangnan_data/ib600_2.csv"
    file2 = "jiangnan_data/ib800_2.csv"
    file3 = "jiangnan_data/ib1000_2.csv"

    file4 = "jiangnan_data/ob600_2.csv"
    file5 = "jiangnan_data/ob800_2.csv"
    file6 = "jiangnan_data/ob1000_2.csv"

    file7 = "jiangnan_data/tb600_2.csv"
    file8 = "jiangnan_data/tb800_2.csv"
    file9 = "jiangnan_data/tb1000_2.csv"

    file10 = "jiangnan_data/n600_3_2.csv"
    file11 = "jiangnan_data/n800_3_2.csv"
    file12 = "jiangnan_data/n1000_3_2.csv"

    n = (500500//2500)*3*3 + 1501500//2500*3
    all_data = np.empty(shape=[0, 2500])
    label = []

    data1 = pd.read_csv(file1, header=None)
    d1 = data1.iloc[:, 0].tolist()
    data2 = pd.read_csv(file2, header=None)
    d2 = data2.iloc[:, 0].tolist()
    data3 = pd.read_csv(file3, header=None)
    d3 = data3.iloc[:, 0].tolist()
    ib_fault = np.array([d1, d2, d3]).reshape(3, len(d1))

    for row in range(ib_fault.shape[0]):
        for col in range(ib_fault.shape[1]//2500):
            all_data = np.append(all_data, ib_fault[row][col*2500: (col+1)*2500].reshape(-1, 2500), axis=0)
            label.append(0)

    data4 = pd.read_csv(file4, header=None)
    d4 = data4.iloc[:, 0].tolist()
    data5 = pd.read_csv(file5, header=None)
    d5 = data5.iloc[:, 0].tolist()
    data6 = pd.read_csv(file6, header=None)
    d6 = data6.iloc[:, 0].tolist()
    ob_fault = np.array([d4, d5, d6]).reshape(3, len(d4))

    for row in range(ob_fault.shape[0]):
        for col in range(ob_fault.shape[1]//2500):
            all_data = np.append(all_data, ob_fault[row][col*2500: (col+1)*2500].reshape(-1, 2500), axis=0)
            label.append(1)

    data7 = pd.read_csv(file7, header=None)
    d7 = data7.iloc[:, 0].tolist()
    data8 = pd.read_csv(file8, header=None)
    d8 = data8.iloc[:, 0].tolist()
    data9 = pd.read_csv(file9, header=None)
    d9 = data9.iloc[:, 0].tolist()
    tb_fault = np.array([d7, d8, d9]).reshape(3, len(d7))

    for row in range(tb_fault.shape[0]):
        for col in range(tb_fault.shape[1]//2500):
            all_data = np.append(all_data, tb_fault[row][col*2500: (col+1)*2500].reshape(-1, 2500), axis=0)
            label.append(2)

    data10 = pd.read_csv(file10, header=None)
    d10 = data10.iloc[:, 0].tolist()
    data11 = pd.read_csv(file11, header=None)
    d11 = data11.iloc[:, 0].tolist()
    data12 = pd.read_csv(file12, header=None)
    d12 = data12.iloc[:, 0].tolist()
    normal = np.array([d10, d11, d12]).reshape(3, len(d10))

    for row in range(normal.shape[0]):
        for col in range(normal.shape[1]//2500):
            all_data = np.append(all_data, normal[row][col*2500: (col+1)*2500].reshape(-1, 2500), axis=0)
            label.append(3)

    label = np.array(label).reshape(-1, 1)

    # get one hot label
    enc = preprocessing.OneHotEncoder()
    enc.fit(label)
    arr = enc.transform(label).toarray()
    sparse_label = np.array(arr, dtype="int64")  # 稀疏标签

    # print(all_data.shape)
    # print(label.shape)
    np.save(os.path.join(npy_save_dir, "all_data.npy"), all_data)
    np.save(os.path.join(npy_save_dir, "label.npy"), sparse_label)


###########
def get_shuffle_feature_and_laebl(feature, label):
    data = np.concatenate((feature, label), axis=1)
    np.random.shuffle(data)
    feature = data[:, 0: 2500]
    label = data[:, 2500:]
    return feature, label


def split_train_val_test(feature_shuffle, sparse_label_shuffle, split_rate=(0.6, 0.2, 0.2)):
    print('----------split train and test----------')
    train_nums = int(feature_shuffle.shape[0]*split_rate[0])
    val_nums   = int(feature_shuffle.shape[0]*split_rate[1])
    test_nums  = int(feature_shuffle.shape[0]*split_rate[2])
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
        return fea_train[:, 0: half_len], fea_val[:, 0: half_len], fea_test[:, 0: half_len]
    else:
        return fea_train, fea_val, fea_test

def get_wavelet_packet_decomposition_feature(train_X, val_X, test_X, fs=50000):
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

def fft_wp_fea(train_X, val_X, test_X, isHalf=True, isPostive=True, fs=50000):
    train_fft, val_fft, test_fft = get_fft_feature(train_X, val_X, test_X, isHalf=True, isPostive=True)
    train_wp, val_wp, test_wp    = get_wavelet_packet_decomposition_feature(train_X, val_X, test_X, fs=50000)
    
    train_X = np.concatenate((train_fft, train_wp), axis=1)
    val_X   = np.concatenate((val_fft, val_wp), axis=1)
    test_X  = np.concatenate((test_fft, test_wp), axis=1)

    return train_X, val_X, test_X


def time_fft_wp_fea(train_X, val_X, test_X, isHalf=True, isPostive=True, fs=50000):
    train_t, val_t, test_t = get_time_fea(train_X, val_X, test_X)
    train_fft, val_fft, test_fft = get_fft_feature(train_X, val_X, test_X, isHalf=True, isPostive=True)
    train_wp, val_wp, test_wp    = get_wavelet_packet_decomposition_feature(train_X, val_X, test_X, fs=50000)
    
    train_X = np.concatenate((train_t, train_fft, train_wp), axis=1)
    val_X   = np.concatenate((val_t, val_fft, val_wp), axis=1)
    test_X  = np.concatenate((test_t, test_fft, test_wp), axis=1)

    return train_X, val_X, test_X




if __name__ == "__main__":
    # generate npy to reduce time c... 
    npy_save_dir = "jiangnan_data"
    csv_to_npy_file(npy_save_dir)



