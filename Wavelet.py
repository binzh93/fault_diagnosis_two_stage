# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:32:26 2016
这是用来做小波能量特征向量提取的一个对象库，主要包含以下：
属性：
1   原始信号的时域序列 org_sound
2   原始信号的频率 fs
3   原始信号小波分解后的高频系数 detail_array
4   原始信号小波分解后的高频系数 approx_array
5   该对象使用的小波分解方式（是分析还是变换）wave_mode
6   该对象使用的小波分解层数 analyze_level
7   该对象用到的小波家族名称 wave_name
方法：
1   wavelet_analyze 小波分析
2   wavelet_tree 小波包分解
3   wprvector 获取小波能量特征向量
4   power_distance 获取能量特征距离向量
"""

import pywt
import numpy as np


class Wavelet_analyze(object):
    def __init__(self, org_sound=[], fs=0):
        self.org_sound = org_sound
        self.fs = fs
        self.approx_array = []
        self.detail_array = []
        self.sig_power_vector = []
        self.sig_power_distance_vector = []
        self.wave_mode = ''
        self.level = np.nan
        self.col_name = []

    """
    wavelet_tree 小波包分解
    """

    def wavelet_tree(self, wavelet='db1', analyze_level=7):
        sound = self.org_sound
        # 进行小波包分解
        wp = pywt.WaveletPacket(data=sound, wavelet=wavelet, mode='sym')
        # 获取小波包分解树的各个节点名称
        node_names = [node.path for node in wp.get_level(analyze_level, 'natural')]
        # 获取 概要部分节点名称 和 细节部分节点名称
        node_names_approx = node_names[:(2 ** analyze_level // 2)]
        node_names_detail = node_names[(2 ** analyze_level // 2):]
        # 分别获取 概要部分的信号 和 细节部分的信号
        approx_array = [pywt.upcoef('a', wp[node].data, wavelet, level=analyze_level) \
                        for node in node_names_approx]
        detail_array = [pywt.upcoef('d', wp[node].data, wavelet, level=analyze_level) \
                        for node in node_names_detail]

        self.approx_array = approx_array
        self.detail_array = detail_array
        self.wave_mode = 'wavelet_tree'
        self.level = analyze_level
#         print(len(self.approx_array), len(self.detail_array))

        self.col_name = [node.path for node in \
                         wp.get_level(analyze_level, 'natural')]
        return approx_array, detail_array

    """
    wavelet_dec 小波分解
    """

    def wavelet_dec(self, wavelet='db1', analyze_level=3):
        sound = self.org_sound
        coeffs = pywt.wavedec(sound, wavelet, level=analyze_level)
        coef_a = coeffs[0]
        coef_d = coeffs[1:]
        approx_array = pywt.upcoef('a', coef_a, wavelet, level=analyze_level)
        approx_array = [approx_array]

        detail_array = list()

        for ii in range(analyze_level):
            detail_array.append(pywt.upcoef('d', coef_d[ii], wavelet, level=(analyze_level - ii)))

        self.approx_array = approx_array
        self.detail_array = detail_array
        self.wave_mode = 'wavelet_dec'
        self.level = analyze_level

        approx = ['a' + str(self.level)]
        detail = ['d' + str(self.level - ii) \
                  for ii in np.arange(self.level)]

        col_name = approx + detail
        self.col_name = col_name
        return approx_array, detail_array

    #    """
    #    col_name() 获取特征向量的每个维度的列名
    #    """
    #    def col_name(self):
    #        if 'wavelet_dec' == self.wave_mode:
    #            approx = ['a' + str(self.level)]
    #            detail = ['b' + str(self.level - ii) \
    #                    for ii in np.arange(self.level)]
    #
    #            col_name = approx + detail
    #            return col_name
    #        else:
    #            if 'wavelet_tree' == self.wave_mode:
    #                col_name =
    #                return col_name
    #            else:
    #                print('请先进行小波分解或小波包分解')
    #                return []

    """
    wprvector 获取小波能量特征向量
    """

    def wprvector(self):

        app_size = len(self.approx_array)
        detail_size = len(self.detail_array)

        if detail_size:
            pass
        else:
            print('请先计算小波系数')
            return

        sig_power_group = np.zeros(app_size + detail_size)
        #   小波分解树中单个节点的能量总和
        ind = 0

        #   求概要信号的能量和
        for ii in np.arange(app_size):
            sig = self.approx_array[ii]
#             print(self.approx_array[ii].shape)
#             print("Ei shape: ",  np.square(sig).shape)
            sig_power_group[ind] = np.sum(np.square(sig))
            ind = ind + 1

        # 求细节信号的能量和
        for ii in np.arange(detail_size):
            sig = self.detail_array[ii]
            sig_power_group[ind] = np.sum(np.square(sig))
            ind = ind + 1

        # 能量特征向量
        sig_power_vector = sig_power_group / np.sum(sig_power_group)
        #print(sig_power_vector)
        self.sig_power_vector = sig_power_vector
        return sig_power_vector

    """
    能量距 power_distance 
    """

    def power_distance(self):

        delta_t = 1 / self.fs
        app_size = len(self.approx_array)
        detail_size = len(self.detail_array)

        if detail_size:
            pass
        else:
            print('请先计算小波系数')
            return
        sig_power_distance_group = np.zeros(app_size + detail_size)
        ind = 0

        #   求概要信号的能量特征距
        for ii in np.arange(app_size):
            sig = self.approx_array[ii]
            # np.arange(len(sig))代表第几个采样点的序列
            # delta_t采样的时间间隔
            # np.square(sig)信号的能量
            temp_1 = np.arange(len(sig)) * delta_t
            temp_2 = np.square(sig) * temp_1

            sig_power_distance_group[ind] = np.sum(temp_2)

            ind = ind + 1

        # 求细节信号的能量特征距
        for ii in np.arange(detail_size):
            sig = self.detail_array[ii]
            # np.arange(len(sig))代表第几个采样点的序列
            # delta_t采样的时间间隔
            # np.square(sig)信号的能量
            temp_1 = np.arange(len(sig)) * delta_t
            temp_2 = np.square(sig) * temp_1

            sig_power_distance_group[ind] = np.sum(temp_2)

            ind = ind + 1

        # 能量距向量
        sig_power_distance_vector = sig_power_distance_group / np.sum(sig_power_distance_group)
        self.sig_power_distance_vector = sig_power_distance_vector
        return sig_power_distance_vector