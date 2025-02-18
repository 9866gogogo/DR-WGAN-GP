import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import pywt  # Continuous Wavelet Transform
import copy
import scipy.stats as st
from scipy.special import comb
import seaborn as sns
from sympy import *
import math
import os
 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict 

from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV 
import kennard_stone as ks 

from sys import stdout
from scipy.signal import savgol_filter

# import seaborn as sns
# import warnings
# warnings.filterwarnings('ignore')
# import  ssl
# ssl._create_default_https_context = ssl._create_unverified_context


from sklearn.decomposition import PCA

# model
from sklearn.cross_decomposition import PLSRegression  
# from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_decomposition import PLSRegression  
from sklearn.linear_model import ElasticNet 
from sklearn.ensemble import RandomForestRegressor  
# from deepforest import CascadeForestRegressor 
from xgboost import XGBRegressor 
from sklearn.svm import SVR 

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pylab import mpl
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
# 设置显示中文字体
# mpl.rcParams["font.sans-serif"] = ["FangSong"]
# plt.rcParams['font.family'] = ['SimSun', 'Times New Roman'] 
mpl.rcParams["font.sans-serif"] = ["SimSun"]
times_italic = FontProperties(family='Times New Roman', style='italic')
times_ = FontProperties(family='Times New Roman')
sim_sun = FontProperties(family='SimSun')
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


# _2023_12_17：
# 自己新配置的环境sklearn版本太新，部分代码出现版本警告遂忽略之...
# 控制scikit-learn版本在1.2.2以下可以保证正常运行
import warnings
# 屏蔽 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
# 恢复警告设置（可选）
# warnings.resetwarnings()

def statistical_characteristic(_data, _show=False):
    """
    计算并返回统计特征
    :param _data: 目标值数据（土壤有机质）
    :param _show: 数据展示
    :return: 返回统计特征的列表[样本量,最小值,最大值,平均值,标准差,方差,变异系数,偏度,峰度]
    """
    total = _data.shape[0]
    min_value = _data.min(axis=0)
    max_value = _data.max(axis=0)
    mean_value = _data.mean(axis=0)
    std_value = _data.std(axis=0)
    var_value = _data.var(axis=0)
    cov_value = std_value/mean_value
    # 偏度
    skew_value = st.skew(_data, axis=0)
    # 峰度
    kurtosis_value = st.kurtosis(_data, axis=0)
    statistic_list = [total, min_value, max_value, mean_value, std_value, var_value, cov_value, skew_value, kurtosis_value]
    # 保留小数点后两位
    if _show:
        print(f'样本量：{total}')
        print(f'最小值/(g.kg^-1)：{min_value:.2f}')
        print(f'最大值/(g.kg^-1)：{max_value:.2f}')
        print(f'平均值/(g.kg^-1)：{mean_value:.2f}')
        print(f'标准差/(g.kg^-1)：{std_value:.2f}')
        print(f'方差/(g.kg^-1)：{var_value:.2f}')
        print(f'变异系数/%：{cov_value:.2f}')
        print(f'偏度/(g.kg^-1)：{skew_value:.2f}')
        print(f'峰度/(g.kg^-1)：{kurtosis_value:.2f}')
    return statistic_list

# def draw_boxplot(*data):
#     plt.figure(figsize=[6,4],dpi=400)
#     p_row = 1
#     p_col = len(data)
#     sample_type = [['总样本','建模集','验证集'],['Whole set','Calibration set','Validation set']]
#     x_labels = []
#     st_info = []
#     for i in range(p_col):
#         st_info.append(statistical_characteristic(data[i],_show=False))
#         x_labels.append(f'{sample_type[0][i]}(n={st_info[i][0]})\n{sample_type[1][i]}')
#     plt.figure(dpi=200)
    
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)

#     plt.boxplot(data, widths=0.2,showmeans=True,labels=x_labels,notch=True)
#     plt.text(x=0.7,y=45,s=f'Mean:{st_info[0][3]:.2f} g·kg$^{{-1}}$\nCV:{st_info[0][6]:.2f}')
#     plt.text(x=1.7,y=45,s=f'Mean:{st_info[1][3]:.2f} g·kg$^{{-1}}$\nCV:{st_info[1][6]:.2f}')
#     plt.text(x=2.7,y=45,s=f'Mean:{st_info[2][3]:.2f} g·kg$^{{-1}}$\nCV:{st_info[2][6]:.2f}')
#     # plt.grid(True)
#     plt.ylabel('土壤有机质含量\n Soil organic matter content(g·kg$^{-1}$)')
#     plt.savefig(f'../Images/数据集分布箱型图.png',bbox_inches = 'tight')
#     plt.show()
def draw_boxplot(*data):
    p_row = 1
    p_col = len(data)
    sample_type = ['总样本','建模集','验证集']
    x_labels = []
    st_info = []
    for i in range(p_col):
        st_info.append(statistical_characteristic(data[i],_show=False))
        x_labels.append(f'{sample_type[i]}(n={st_info[i][0]})')
    
    # fig, ax = plt.subplots(figsize=[6, 4],dpi=400)
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    bplot = plt.boxplot(data, widths=0.2,showmeans=True,labels=x_labels,notch=True,patch_artist=True)
    plt.text(x=0.7,y=47,s=f'Mean:{st_info[0][3]:.2f} g·kg$^{{-1}}$\nCV:{st_info[0][6]:.2f}',fontdict={'fontsize': 10,'family': 'Times New Roman'})
    plt.text(x=1.7,y=47,s=f'Mean:{st_info[1][3]:.2f} g·kg$^{{-1}}$\nCV:{st_info[1][6]:.2f}',fontdict={'fontsize': 10,'family': 'Times New Roman'})
    plt.text(x=2.7,y=47,s=f'Mean:{st_info[2][3]:.2f} g·kg$^{{-1}}$\nCV:{st_info[2][6]:.2f}',fontdict={'fontsize': 10,'family': 'Times New Roman'})
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(True,linestyle = '--',alpha=0.7)
    plt.ylabel('土壤有机质含量(g·kg$^{-1}$)', fontproperties=sim_sun)
    # plt.savefig(f'../Images/数据集分布箱型图.png',bbox_inches = 'tight')
    plt.show()

def show_hyperspectral_image(_data, title=None, x_label_start=0, sample_interval=10):
    """
    展示预处理后的数据图像
    :param _data: 原始或预处理后的光谱数据
    :param title: 图像文件的标题
    :param x_label_start: 光谱图像的起始波段值
    :param sample_interval: 光谱图像的重采样间隔
    :return: 显示并保存图形至指定目录
    """
    y = _data
    x = range(0, _data.shape[1])

    # 默认处理采样间隔10nm的光谱数据
    axis_x_label = range(x_label_start, y.shape[1] * sample_interval + x_label_start, sample_interval)

    fig, ax = plt.subplots(figsize=[6, 4],dpi=400)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for i in range(0, y.shape[0]):
        plt.plot(x, y[i])
    
    # 控制标记间隔
    if sample_interval == 10:
        xticks_interval = 20
    elif sample_interval == 1:
        xticks_interval = 200 
    plt.xticks(x[::xticks_interval], axis_x_label[::xticks_interval], rotation=0)
    plt.xlabel('Wavelength/nm', fontsize=13, fontproperties=times_)
    plt.ylabel('Reflectance', fontsize=13, fontproperties=times_)
    plt.title(title, fontsize=15, fontdict={'fontsize': 18,'family': 'SimSun'})
    plt.grid(linestyle = '--',alpha=0.7)
    plt.show()

def SG(data, w=11, p=2):
    """
    Savitzky-Golay平滑滤波
    :param data:
    :param w:
    :param p:
    :return:
    """
    return signal.savgol_filter(data, w, p)


def CT(data):
    """
    均值中心化
    :param data:
    :return:
    """
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


def min_max_scaler(_data):
    """
    归一化
    :param _data:
    :return:
    """
    return MinMaxScaler().fit_transform(_data)


def standardization(_data):
    """
    标准化
    :param _data:
    :return:
    """
    return StandardScaler().fit_transform(_data)


def SNV(data):
    """
    标准正态变换
    :param data:
    :return:
    """
    m = data.shape[0]
    n = data.shape[1]
    # print(m, n)
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i]) for j in range(n)] for i in range(m)]
    return np.array(data_snv)


def ln_R(data):
    """
    对数倒数
    :param data:
    :return:
    """    
    return np.log10((1 / data))


def D1(data):
    """
    一阶导数
    :param data:
    :return:
    """
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


def D2(data):
    """
    二阶导数
    :param data:
    :return:
    """
    data = copy.deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    temp2 = (pd.DataFrame(data)).diff(axis=1)
    temp3 = np.delete(temp2.values, 0, axis=1)
    temp4 = (pd.DataFrame(temp3)).diff(axis=1)
    spec_D2 = np.delete(temp4.values, 0, axis=1)
    return spec_D2


def MSC(data):
    """
    多元散射校正
    :param data:
    :return:
    """
    n, p = data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(data, axis=0)

    # 线性拟合
    for i in range(n):
        y = data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc

# 趋势校正(DT)
def DT(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after DT :(n_samples, n_features)
    """
    x = np.asarray(range(350, data.shape[1]*10+350,10), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)
    return out



def CWT(spectra_data):
    """
    连续小波变
    :param spectra_data:
    :return:
    """
    num_samples = spectra_data.shape[0]

    # CWT参数设置
    wavelet = 'gaus4'  # 小波类型，可以根据需要选择
    scales = np.arange(1, 10)  # 尺度范围

    # 对每个光谱数据样本进行CWT
    cwt_results = []
    for sample in spectra_data:
        coefs, frequencies = pywt.cwt(sample, scales, wavelet)
        cwt_results.append(coefs)

    # 可视化CWT结果
    for i in range(num_samples):
        plt.figure()
        # plt.imshow(np.abs(cwt_results[i]), extent=[wavelengths.min(), wavelengths.max(), scales.min(), scales.max()],
        #            aspect='auto', cmap='inferno')

        plt.imshow(np.abs(cwt_results[i]),
                   extent=[400, 2400, scales.min(), scales.max()],
                   aspect='auto', cmap='inferno')

        plt.title(f"CWT for Sample {i + 1}")
        plt.xlabel('Wavelength')
        plt.ylabel('Scale')
        plt.colorbar()
        plt.show()


def wavelet_denoising(_data, wavelet_bases='db4', level=0, filtering_threshold=0.1):
    """
    小波降噪（WD）
    :param _data: 原始光谱数据
    :param wavelet_bases: 小波基
    :param level: 小波分解层数 默认则会自动根据数据长度计算分解层数
    :param filtering_threshold: 设置降噪阈值，会将(小波分解后每层的最大值 * 阈值) 视作为噪声
    :return: 返回小波降噪后的数据，（会自动去除奇数原始数据去噪后末尾产生新特征）
    """
    wavelet_basis = wavelet_bases

    if level == 0:
        # 根据数据长度计算分解层数
        w = pywt.Wavelet(wavelet_basis)
        max_level = pywt.dwt_max_level(_data.shape[1], w.dec_len)
    elif level > 0:
        # 自定义分解层数
        max_level = level

    threshold = filtering_threshold

    # 去噪前后信号的长度会有所变化，若原始信号样本长度为奇数，则去噪后长度加1，若原始长度为偶数，则去噪后长度不变
    if _data.shape[1] % 2 == 0:
        signalrec_matrix = np.ones([_data.shape[0], _data.shape[1]])
    else:
        signalrec_matrix = np.ones([_data.shape[0], _data.shape[1] + 1])

    for j in range(_data.shape[0]):
        signal = _data[j]
        coeffs = pywt.wavedec(signal, wavelet_basis, level=max_level)  # 将信号进行小波分解
        for i in range(1, len(coeffs)):
            # 将噪声滤波
            # pywt.threshold 方法有三种阈值模式：硬阈值、软阈值、软硬结合
            # 默认为soft 软阈值模式，其调用格式为
            # pywt.threshold(signal, threshold_value, mode='soft', substitute=0)
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        # 重建信号
        len(pywt.waverec(coeffs, wavelet_basis))
        signalrec_matrix[j] = pywt.waverec(coeffs, wavelet_basis)

    # 应该计算去噪后信号的第一个元素到倒数第二个元素这个子序列与原始信号之间的差值，也即舍去因去噪而多出的最后一个值。
    # 如果使用去噪后的第二个元素到最后一个元素与原始信号的差，则可能出现偏差较大的情况。
    if _data.shape[1] % 2 == 0:
        return signalrec_matrix
    else:
        return signalrec_matrix[:, :-1]

# def correlation_analysis(_data, target_value, alpha_sw=False, alpha=0.01, r_sw=False, r=0.0):
#     """
#     计算数据列与目标列的相关性和显著性
#     :param _data:光谱数据
#     :param target_value:计算与_data中间相关性的目标列
#     :param alpha_sw:开启显著性检验，默认为false
#     :param alpha:设置显著性检验水平
#     :param r_sw:开启相关性筛选，默认为false
#     :param r:设置绝对值之后的最低相关性
#     :return:返回相关性系数矩阵，p_value矩阵，原始数据通过显著性水平的矩阵，原始数据通过相关性系数检验的矩阵
#     """
#     x = _data
#     correlation_matrix = np.ones(x.shape[1])
#     p_value_matrix = np.ones(x.shape[1])
#     alpha_index = []
#     r_index = []
#     for i in range(x.shape[1]):
#         correlation, p_value = pearsonr(x[:, i], target_value)
#         correlation_matrix[i] = correlation
#         p_value_matrix[i] = p_value
#         if alpha_sw == True and abs(p_value) < alpha:
#             # print(f'Attribute {i}:')
#             # print(f'p-value: {p_value}')
#             alpha_index.append(i)
#         if r_sw == True and abs(correlation) > r:
#             # print(f'Attribute {i}:')
#             # print(f'Correlation coefficient: {correlation}')
#             r_index.append(i)
#     # alpha_matrix = x[:, alpha_index]
#     # r_matrix = x[:, r_index]
#     # print(f'共有{len(alpha_index)}个通过显著性水平0.01检验的波段：{[i*10+350 for i in alpha_index]}') # 过假设检验的波段默认采样间隔为10
#     # print(f'共有{len(alpha_index)}个通过显著性水平0.01检验的波段：{[i*1+350 for i in alpha_index]}') # 过假设检验的波段默认采样间隔为1
#     print(f'共有{len(alpha_index)}个通过显著性水平0.01检验的波段')
#     print(f"其中相关性绝对值过{r}的波段有{len(r_index)}个")
#     return correlation_matrix, p_value_matrix, alpha_index, r_index

# def correlation_analysis_show(correlation_matrix, p_value_matrix,title=None):
#     interval = 10 # 采样间隔
#     star = 400 # 起始波段
#     end = correlation_matrix.shape[0]*interval+star # 结束波段
#     bands_ticks = np.array(range(star,end,interval))
#     abs_corr_matrix = np.abs(correlation_matrix) # 取相关性的绝对值
#     plt.figure(figsize=(11,4),dpi=400,constrained_layout=True)
    
#     plt.subplot(121)
#     ax=plt.gca()  #gca:get current axis得到当前轴
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     plt.plot(bands_ticks,abs_corr_matrix,'k-',linewidth=0.9,label='相关性系数绝对值')
#     plt.axhline(y=0.5,linewidth=0.8,xmin=0,xmax=0.95,ls='--',color='r',label='相关性系数绝对值=0.5水平线')
#     cor_max = abs_corr_matrix.max() # 获取相关性绝对值的最大值
#     cor_max_index = bands_ticks[np.argwhere(abs_corr_matrix==cor_max)[0]]
#     plt.text(cor_max_index-200,cor_max,f'光谱波段:{cor_max_index[0]},\n最大相关性系数绝对值:{cor_max:.3f}\n',color='k')
#     plt.plot(cor_max_index,cor_max,'x', color = 'r')
#     plt.vlines(cor_max_index, 0.0, cor_max, linestyles='dashed', colors='red',linewidth=0.6) # 标记垂线
#     plt.xlabel("波长\nWave length/nm",fontsize=11)
#     plt.ylabel("相关性系数\nThe correlation coefficient",fontsize=11)
#     plt.ylim([0.0, 1.0])
#     plt.grid(linestyle = '--',alpha=0.7)
#     plt.legend()
#     plt.title(f'{title}')

def correlation_analysis(_data, target_value, alpha=0.01, r=0.0):
    """
    计算数据列与目标列的相关性和显著性
    :param _data:光谱数据
    :param target_value:计算与_data中间相关性的目标列
    :param alpha_sw:开启显著性检验，默认为false
    :param alpha:设置显著性检验水平
    :param r_sw:开启相关性筛选，默认为false
    :param r:设置绝对值之后的最低相关性
    :return:返回相关性系数矩阵，p_value矩阵，原始数据通过显著性水平的矩阵，原始数据通过相关性系数检验的矩阵
    """
    x = _data
    correlation_matrix = np.ones(x.shape[1])
    p_value_matrix = np.ones(x.shape[1])
    alpha_index = []
    r_index = []
    for i in range(x.shape[1]):
        correlation, p_value = pearsonr(x[:, i], target_value)
        correlation_matrix[i] = correlation
        p_value_matrix[i] = p_value
        if abs(p_value) < alpha:
            alpha_index.append(i)
        if abs(correlation) > r:
            r_index.append(i)
    # alpha_matrix = x[:, alpha_index]
    # r_matrix = x[:, r_index]
    # print(f'共有{len(alpha_index)}个通过显著性水平0.01检验的波段：{[i*10+350 for i in alpha_index]}') # 过假设检验的波段默认采样间隔为10
    # print(f'共有{len(alpha_index)}个通过显著性水平0.01检验的波段：{[i*1+350 for i in alpha_index]}') # 过假设检验的波段默认采样间隔为1
    print(f'共有{len(alpha_index)}个通过显著性水平{alpha}检验的波段')
    print(f"其中相关性绝对值过{r}的波段有{len(r_index)}个")
    return correlation_matrix, p_value_matrix, alpha_index, r_index

def correlation_analysis_show(correlation_matrix, p_value_matrix,title=None):
    interval = 10 # 采样间隔
    star = 400 # 起始波段
    end = correlation_matrix.shape[0]*interval+star # 结束波段
    bands_ticks = np.array(range(star,end,interval))
    abs_corr_matrix = np.abs(correlation_matrix) # 取相关性的绝对值
    plt.figure(figsize=(11,4),dpi=400,constrained_layout=True)
    
    plt.subplot(121)
    ax=plt.gca()  #gca:get current axis得到当前轴
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.plot(bands_ticks,abs_corr_matrix,'k-',linewidth=0.9,label='相关性系数绝对值')
    plt.axhline(y=0.5,linewidth=0.8,xmin=0,xmax=0.95,ls='--',color='r',label='相关性系数绝对值=0.5水平线')
    cor_max = abs_corr_matrix.max() # 获取相关性绝对值的最大值
    cor_max_index = bands_ticks[np.argwhere(abs_corr_matrix==cor_max)[0]]
    # plt.text(cor_max_index-200,cor_max,f'光谱波段:{cor_max_index[0]},\n最大相关性系数绝对值:{cor_max:.3f}\n',color='k')
    plt.text(1600,0.6,f'光谱波段:{cor_max_index[0]},\n最大相关性系数绝对值:{cor_max:.3f}\n',color='k')
    plt.plot(cor_max_index,cor_max,'x', color = 'r')
    plt.vlines(cor_max_index, 0.0, cor_max, linestyles='dashed', colors='red',linewidth=0.6) # 标记垂线
    plt.xlabel("波长(nm)",fontsize=11)
    plt.ylabel("相关性系数",fontsize=11)
    plt.ylim([0.0, 1.0])
    plt.grid(linestyle = '--',alpha=0.7)
    plt.legend()
    plt.title(f'{title}', fontdict={'fontsize': 13,'family': 'Times New Roman'})
    
    plt.subplot(122)
    ax=plt.gca()  #gca:get current axis得到当前轴
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.plot(bands_ticks,p_value_matrix,'k-',linewidth=0.9,label='p-value')
    plt.axhline(y=0.05,linewidth=0.8,xmin=0.05,xmax=0.98,ls='--',color='r',label='显著性水平0.05')
    plt.xlabel("波长(nm)",fontsize=11)
    plt.ylabel("P-值",fontsize=11)
    plt.legend(loc="upper right")  
    plt.title(f'{title}', fontdict={'fontsize': 13,'family': 'Times New Roman'})
    plt.grid(linestyle = '--',alpha=0.7)

    plt.savefig(f'../Images/{title}correlation_analysis.png',bbox_inches = 'tight')
    plt.show()
    
    # plt.subplot(122)
    # ax=plt.gca()  #gca:get current axis得到当前轴
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # plt.plot(bands_ticks,p_value_matrix,'k-',linewidth=0.9,label='p-value')
    # plt.axhline(y=0.01,linewidth=0.8,xmin=0.05,xmax=0.98,ls='--',color='r',label='显著性水平0.01')
    # plt.xlabel("波长\nWave length/nm",fontsize=11)
    # plt.ylabel("P-值\nThe p-value",fontsize=11)
    # plt.legend(loc="upper right")  
    # plt.title(f'{title}')
    # plt.grid(linestyle = '--',alpha=0.7)

    # plt.savefig(f'../Images/{title}correlation_analysis.png',bbox_inches = 'tight')
    # plt.show()

def PC_Cross_Validation(X, y, pc, cv):
    '''
        x :光谱矩阵 nxm
        y :浓度阵 （化学值）
        pc:最大主成分数
        cv:交叉验证数量
    return :
        RMSECV:各主成分数对应的RMSECV
        PRESS :各主成分数对应的PRESS
        rindex:最佳主成分数
    '''
    kf = KFold(n_splits=cv)
    RMSECV = []
    for i in range(pc):
        RMSE = []
        for train_index, test_index in kf.split(X):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, y_train)
            y_predict = pls.predict(x_test)
            RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
        RMSE_mean = np.mean(RMSE)
        RMSECV.append(RMSE_mean)
    rindex = np.argmin(RMSECV)
    return RMSECV, rindex


def Cross_Validation(X, y, pc, cv):
    '''
     x :光谱数据
     y :目标值
     pc:最大主成分数
     cv:交叉验证数量
     return :
            RMSECV:各主成分数对应的RMSECV
    '''
    kf = KFold(n_splits=cv)
    RMSE = []
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pls = PLSRegression(n_components=pc)
        pls.fit(x_train, y_train)
        y_predict = pls.predict(x_test)
        RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
    RMSE_mean = np.mean(RMSE)
    return RMSE_mean


def CARS_Cloud(X, y, N=50, f=20, cv=10):
    p = 0.8
    m, n = X.shape
    u = np.power((n/2), (1/(N-1)))
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)
    # val_num = m - cal_num
    b2 = np.arange(n)
    x = copy.deepcopy(X)
    D = np.vstack((np.array(b2).reshape(1, -1), X))
    WaveData = []
    # Coeff = []
    WaveNum =[]
    RMSECV = []
    r = []
    for i in range(1, N+1):
        r.append(u*np.exp(-1*k*i))
        wave_num = int(np.round(r[i-1]*n))
        WaveNum = np.hstack((WaveNum, wave_num))
        cal_index = np.random.choice    \
            (np.arange(m), size=int(cal_num), replace=False)
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        #xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1,-1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData  = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        # cb = coeff[:wave_num]
        #
        # if wnum > 0:
        #     cb = np.vstack((cb, np.full((wnum, 1), -1)))
        # if len(Coeff) == 0:
        #     Coeff = copy.deepcopy(cb)
        # else:
        #     Coeff = np.hstack((Coeff, cb))
        rmsecv, rindex = PC_Cross_Validation(xcal, ycal, f, cv)
        RMSECV.append(Cross_Validation(xcal, ycal, rindex+1, cv))
    # CoeffData = Coeff.T

    WAVE = []
    # COEFF = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        # cd = CoeffData[i, :]
        WD = np.ones((len(wd)))
        # CO = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
                # CO[j] = 0
            else:
                WD[j] = wd[ind[0]]
                # CO[j] = cd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
        # if len(COEFF) == 0:
        #     COEFF = copy.deepcopy(CO)
        # else:
        #     COEFF = np.vstack((WAVE, CO.reshape(1, -1)))

    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    boindex = np.where(Optimal != 0)
    OptWave = boindex[0]

    #fig = plt.figure()
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 11
    plt.figure(figsize=(7,5),dpi=400, constrained_layout=True)
    plt.subplot(211)
    #plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('被选择的波长数量', fontsize=fonts, fontproperties=sim_sun)
    plt.title(f'最佳迭代次数：{str(MinIndex)}次', fontdict={'fontsize': 11,'family': 'SimSun'})
    plt.plot(np.arange(N), WaveNum,'k',linewidth=0.7)
    plt.grid(linestyle = '--',alpha=0.7)

    plt.subplot(212)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts, fontproperties=sim_sun)
    plt.ylabel('交叉验证误差', fontsize=fonts, fontproperties=sim_sun)
    plt.plot(np.arange(N), RMSECV,'k',linewidth=0.7)
    plt.grid(linestyle = '--',alpha=0.7)

    # # plt.subplot(313)
    # # plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    # # plt.ylabel('各变量系数值', fontsize=fonts)
    # # plt.plot(COEFF)
    # #plt.vlines(MinIndex, -1e3, 1e3, colors='r')
    plt.savefig(f'../Images/CARS_Cloud.jpg',bbox_inches = 'tight')
    plt.show()
    
    return OptWave

# 变量重要性分析，变量对y的影响程度，一般认为大于1是有影响的
def Cal_VIP(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    # s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)  # @表示矩阵相乘
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
    return vips

def draw_scatter(true, pred, title=None):
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel('Actual Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(title)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(45, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black')
    plt.grid(linestyle = '--',alpha=0.7)
    # plt.savefig(f'../Program/Images/{title}.png')    
    plt.show()

def evaluate_model(model,x_train,y_train,x_test,y_test,model_name=None):
    model.fit(x_train,y_train)
    
    plt.figure(figsize=(7,5),dpi=400, constrained_layout=True)
    plt.subplot(221)
    plt.plot(range(1,len(y_train)+1),y_train,color = 'r',label='实测值')
    plt.plot(range(1,len(y_train)+1),model.predict(x_train),color = 'b',label='估测值')
    plt.xlabel(f'建模集\nCalibration set({x_train.shape[0]})', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''实测/估测值(g/kg$^{-1}$)
    Actual/Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.legend(loc='upper right',frameon=False)
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(222)
    true = y_train
    pred = model.predict(x_train)
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel(r'''实测值(g/kg$^{-1}$)
    Actual values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''估测值(g/kg$^{-1}$)
    Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(30, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black', fontdict={'fontsize': 18,'style': 'italic','family': 'Times New Roman'})
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(223)
    plt.plot(range(1,len(y_test)+1),y_test,color = 'r',label='实测值')
    plt.plot(range(1,len(y_test)+1),model.predict(x_test),color = 'b',label='估测值')
    plt.xlabel(f'验证集\nPrediction set({x_test.shape[0]})', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''实测/估测值(g/kg$^{-1}$)
    Actual/Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.legend(loc='upper right',frameon=False)
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(224)
    true = y_test
    pred = model.predict(x_test)
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel(r'''实测值(g/kg$^{-1}$)
    Actual values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''估测值(g/kg$^{-1}$)
    Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    # plt.title(title)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(24, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black', fontdict={'fontsize': 18,'style': 'italic','family': 'Times New Roman'})
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.savefig(f'../../Images/20240911/{model_name}.png',bbox_inches = 'tight')
    pass

def evaluate_ANN_model(model,x_train,y_train,x_test,y_test,model_name=None):
    
    plt.figure(figsize=(7,5),dpi=400, constrained_layout=True)
    plt.subplot(221)
    plt.plot(range(1,len(y_train)+1),y_train,color = 'r',label='实测值')
    plt.plot(range(1,len(y_train)+1),model.predict(x_train),color = 'b',label='估测值')
    plt.xlabel(f'建模集\nCalibration set({x_train.shape[0]})', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''实测/估测值(g/kg$^{-1}$)
    Actual/Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.legend(loc='upper right',frameon=False)
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(222)
    true = y_train
    pred = model.predict(x_train)
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel(r'''实测值(g/kg$^{-1}$)
    Actual values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''估测值(g/kg$^{-1}$)
    Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(30, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black', fontdict={'fontsize': 18,'style': 'italic','family': 'Times New Roman'})
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(223)
    plt.plot(range(1,len(y_test)+1),y_test,color = 'r',label='实测值')
    plt.plot(range(1,len(y_test)+1),model.predict(x_test),color = 'b',label='估测值')
    plt.xlabel(f'验证集\nPrediction set({x_test.shape[0]})', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''实测/估测值(g/kg$^{-1}$)
    Actual/Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.legend(loc='upper right',frameon=False)
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(224)
    true = y_test
    pred = model.predict(x_test)
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel(r'''实测值(g/kg$^{-1}$)
    Actual values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''估测值(g/kg$^{-1}$)
    Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    # plt.title(title)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(24, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black', fontdict={'fontsize': 18,'style': 'italic','family': 'Times New Roman'})
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    # plt.savefig(f'../Images/{model_name}.png',bbox_inches = 'tight')
    pass
    
def pls_variable_selection(X, y, max_comp):
    # max_comp:允许PLS中主成分的最大数量
    mse = np.zeros((max_comp,X.shape[1]))
    for i in range(max_comp):
        pls1 = PLSRegression(n_components=i+1)
        pls1.fit(X, y)
        sorted_ind = np.argsort(np.abs(pls1.coef_[:,0])) # 返回PLS系数数组值从小到大的索引值
        Xc = X[:,sorted_ind] # 将波段按照PLS系数从小到大进行排序
        for j in range(Xc.shape[1]-(i+1)): # 一次丢弃一个波长的排序光谱
            pls2 = PLSRegression(n_components=i+1) # 回归
            pls2.fit(Xc[:, j:], y)
            y_cv = cross_val_predict(pls2, Xc[:, j:], y, cv=5) # 计算MSE交叉验证
            mse[i,j] = mean_squared_error(y, y_cv)
                                         
        comp = 100*(i+1)/(max_comp) # 循环中实时更新进度，显示当前完成的百分比
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # 搜索mse的全局最小值（不包括零）,返回其对应的 max_comp 值和 波长数量
    mseminx,mseminy = np.where(mse==np.min(mse[np.nonzero(mse)]))

    print("Optimised number of PLS components: ", mseminx[0]+1)
    print("Wavelengths to be discarded ",mseminy[0])
    print('Optimised RMSEP ', math.sqrt(mse[mseminx,mseminy][0]))
    stdout.write("\n")

    # 使用最佳主成分回归并生成PLS系数
    pls = PLSRegression(n_components=mseminx[0]+1)
    pls.fit(X, y)      
    sorted_ind = np.argsort(np.abs(pls.coef_[:,0]))
    Xc = X[:,sorted_ind]

    # 返回选择的最佳波段,主成分的最佳数量，剔除的波长数，PLS系数从小到大的索引值
    return(Xc[:,mseminy[0]:],mseminx[0]+1,mseminy[0], sorted_ind)


def simple_pls_cv(X, y, n_comp):
 
    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, y)
    y_c = pls.predict(X)
 
    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)    
 
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
 
    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
 
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
 
    # Plot regression 
 
    z = np.polyfit(y, y_cv, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(y_cv, y, c='red', edgecolors='k')
        ax.plot(z[1]+z[0]*y, y, c='blue', linewidth=1)
        ax.plot(y, y, color='green', linewidth=1)
        plt.title('$R^{2}$ (CV): '+str(score_cv))
        plt.xlabel('Predicted $^{\circ}$Brix')
        plt.ylabel('Measured $^{\circ}$Brix')
        
        plt.show()

def glfdiff(data, gam):
    t = range(data.shape[1])
    result = np.zeros(data.shape)
    for item in range(data.shape[0]):
        y = data[item]
        n = len(y)
        h = t[1] - t[0] 
        w = np.zeros(n)
        w[0] = 1
        a0 = y[0]
        dy = np.zeros(len(t))
        
        if a0 != 0 and gam > 0:
            dy[0] = np.sign(a0) * np.inf
        
        for j in range(1, n):
            w[j] = w[j-1] * (1 - (gam + 1) / j)
        
        for i in range(len(t)):
            dy[i] = np.dot(w[:i+1], y[i::-1]) / h**gam
        result[item] = dy
    return result

def evaluate_CNN_model(model,x_train,y_train,x_test,y_test,model_name=None):

    true = y_train
    true = true.cpu().numpy()
    pred = model(x_train.unsqueeze(1))
    pred = pred.cpu().detach().numpy() 
    
    plt.figure(figsize=(7,5),dpi=400, constrained_layout=True)
    plt.subplot(221)
    plt.plot(range(1,len(true)+1),true,color = 'r',label='实测值')
    plt.plot(range(1,len(true)+1),pred,color = 'b',label='估测值')
    plt.xlabel(f'建模集\nCalibration set({x_train.shape[0]})', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''实测/估测值(g/kg$^{-1}$)
    Actual/Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.legend(loc='upper right',frameon=False)
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(222)
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel(r'''实测值(g/kg$^{-1}$)
    Actual values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''估测值(g/kg$^{-1}$)
    Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(30, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black', fontdict={'fontsize': 18,'style': 'italic','family': 'Times New Roman'})
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})

    true = y_test
    true = true.cpu().numpy()
    pred = model(x_test.unsqueeze(1))
    pred = pred.cpu().detach().numpy() 
    
    plt.subplot(223)
    plt.plot(range(1,len(true)+1),true,color = 'r',label='实测值')
    plt.plot(range(1,len(true)+1),pred,color = 'b',label='估测值')
    plt.xlabel(f'验证集\nPrediction set({x_test.shape[0]})', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''实测/估测值(g/kg$^{-1}$)
    Actual/Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.legend(loc='upper right',frameon=False)
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.subplot(224)
    plt.scatter(true, pred,c='black')
    x = np.linspace(0, max(true), 100)
    plt.plot(x, x, linestyle='--', color='black',)
    plt.xlabel(r'''实测值(g/kg$^{-1}$)
    Actual values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    plt.ylabel(r'''估测值(g/kg$^{-1}$)
    Estimated values(g/kg$^{-1}$)''', fontsize=12, fontproperties=sim_sun)
    # plt.title(title)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mse)
    rpd = np.std(true)/rmse
    plt.text(24, 0.05, f'R$^2$={r2:.2f}\nRPD={rpd:.2f}\nRMSE={rmse:.2f}', fontsize=12, color='black', fontdict={'fontsize': 18,'style': 'italic','family': 'Times New Roman'})
    plt.title(model_name, fontdict={'fontsize': 11,'family': 'Times New Roman'})
    
    plt.savefig(f'../../Images/20241126/{model_name}.png',bbox_inches = 'tight')
    pass