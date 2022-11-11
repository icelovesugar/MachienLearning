import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import time
import os
warnings.filterwarnings('ignore')
import wavio
import librosa
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed,Bidirectional
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
import sklearn

filename_list = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left',
                 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree',
                 'two', 'up', 'wow', 'yes', 'zero']

features, labels = np.empty((0, 68, 44)), np.empty(0)
print(features.ndim)
for i in range(3):  # 0-30 三十个文件夹
    print(filename_list[i])
    filePath = "D:\\machinelearn\\train\\%s" % filename_list[i]
    fl = os.listdir(filePath)
    print(len(fl))  # bed:1537
    for j in range(100):  # 0-fl 每个文件夹中的文件个数 len(fl)
        wavpath = filePath + '\\' + fl[j]
        print(fl[j])
        sig, sr = sf.read(wavpath)  # sig 信号值 sr 采样率
        # print(sig.shape)
        # print(sr)
        # 不够长度的信号进行补0
        x = np.pad(sig, (0, 22050-sig.shape[0]), 'constant')
        print(x.shape)
        # 提取信号的特征并进行标准化
        a = librosa.feature.zero_crossing_rate(x, sr)  # (1,44)
        b = librosa.feature.spectral_centroid(x, sr=sr)[0]  # (44,)

        a = np.vstack((a, b))  # (2,44)
        b = librosa.feature.chroma_stft(x, sr)  # (12,44)

        a = np.vstack((a, b))  # (14,44)
        b = librosa.feature.spectral_contrast(x, sr)  # (7,44)

        a = np.vstack((a, b))  # (21,44)
        b = librosa.feature.spectral_bandwidth(x, sr)  # (1,44)

        a = np.vstack((a, b))  # (22,44)
        b = librosa.feature.tonnetz(x, sr)  # (6,44)

        a = np.vstack((a, b))  # (28,44)
        b = librosa.feature.mfcc(x, sr, n_mfcc=40)  # (40,44)

        a = np.concatenate((a, b))  # (68,44)

        norm_a = sklearn.preprocessing.scale(a, axis=1)  # 数据标准化 按列减去均值再除以方差
        features = np.append(features, norm_a[None], axis=0)
        labels = np.append(labels, int(i))
        print(features.shape)  # (58000,68,44)
    print('*****%s*****' % i)
print("train over!")

# 测试集
X_test = np.empty((0, 68, 44))
filePath = "D:\\machinelearn\\test"
fl = os.listdir(filePath)
print(len(fl))  # 6837个wav文件
for j in range(100):  # len(fl)
    wavpath = filePath + '\\' + fl[j]
    print(fl[j])
    sig, sr = sf.read(wavpath)  # sig 信号值 sr 采样率
    # print(sig.shape)
    # print(sr)
    # 不够长度的信号进行补0
    x = np.pad(sig, (0, 22050 - sig.shape[0]), 'constant')
    print(x.shape)
    # 提取信号的特征并进行标准化
    a = librosa.feature.zero_crossing_rate(x, sr)  # (1,44)
    b = librosa.feature.spectral_centroid(x, sr=sr)[0]  # (44,)

    a = np.vstack((a, b))  # (2,44)
    b = librosa.feature.chroma_stft(x, sr)  # (12,44)

    a = np.vstack((a, b))  # (14,44)
    b = librosa.feature.spectral_contrast(x, sr)  # (7,44)

    a = np.vstack((a, b))  # (21,44)
    b = librosa.feature.spectral_bandwidth(x, sr)  # (1,44)

    a = np.vstack((a, b))  # (22,44)
    b = librosa.feature.tonnetz(x, sr)  # (6,44)

    a = np.vstack((a, b))  # (28,44)
    b = librosa.feature.mfcc(x, sr, n_mfcc=40)  # (40,44)

    a = np.concatenate((a, b))  # (68,44)

    norm_a = sklearn.preprocessing.scale(a, axis=1)  # 数据标准化 按列减去均值再除以方差
    X_test = np.append(X_test, a[None], axis=0)
    print(X_test.shape)  # (6835,68,44)
print("test over!")

# CNN多分类
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)  # 为X_test再多加一个维度
print(X_test.shape)
nfold = 5
kf = KFold(n_splits=nfold, shuffle=True, random_state=2022)  # 用于分割训练集
print(len(X_test))  # 100
prediction1 = np.zeros((len(X_test), 30))  # 100*30的数组
i = 0
print(features.shape)  # (100,40,44)
# 这是把全部的训练数据拆分成了K份，K-1 份测试，1份验证, train_index:0-100 valid_index:100个数随机取30个
for train_index, valid_index in kf.split(features, labels):
    # print(features.shape)  # (100,40,44)
    print("\nFold {}".format(i + 1))
    train_x, train_y = features[train_index], labels[train_index]
    # print(train_x)
    # print(valid_index)
    val_x, val_y = features[valid_index], labels[valid_index]
    # print(val_x)
    # 给train_x,val_x再加一个维度
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], val_x.shape[2], 1)
    print(train_x.shape)  # (240,68,44,1)
    print(val_x.shape)  # (60,68,44,1)
    print(train_y.shape)  # (240,)
    print(val_y.shape)  # (60,)
    # 将类别向量转换为二进制（只有0和1）的矩阵类型表示
    train_y = to_categorical(train_y, 30)
    val_y = to_categorical(val_y, 30)
    print(train_y.shape)  # (80,30)
    print(val_y.shape)  # (20,30)

    model = Sequential()  # 用于构建CNN神经网络
    model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=train_x.shape[1:]))  # 添加一个带有32个 3*3的过滤器
    model.add(MaxPooling2D((2, 2)))  # 添加一个2*2的最大池化层
    model.add(Dropout(0.25))  # Dropout
    model.add(Convolution2D(32, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())  # 展平层
    model.add(Dense(30, activation='softmax'))  # 全连接层 30:输出维度
    model.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])  # 选择优化器optimizer:Adam和指定损失函数loss:categorical_crossentropy
    model.summary(line_length=80)
    history = model.fit(train_x, train_y, epochs=35, batch_size=32, validation_data=(val_x, val_y))  # 调用fit函数将数据提供给模型
    model.save("cnn.h5")
    print(X_test.shape)  # (100,68,44,1)
    y1 = model.predict(X_test, verbose=1)  # 预测 返回值为每个测试集预测的30个类别的概率
    print(y1.shape)  # (100,30)
    prediction1 +=((y1)) / nfold
    i += 1
# print(prediction1)
# 提取预测结果并写入csv文件
y_pred = [list(x).index(max(x)) for x in prediction1]
print(len(y_pred))  # 100
sub = pd.read_csv('D:\\machinelearn\\submission.csv', engine='python')
cnt = 0
result = [0 for i in range(6835)]
for i in range(6835):  # 6835
    ss = sub.iloc[i]['file_name']  # 获取i行file_name列
    for j in range(6835):  # 6835
        if fl[j] == ss:
            result[i] = y_pred[j]
            cnt = cnt+1
print(cnt)
result1 = []
for i in range(len(result)):
    result1.append(filename_list[result[i]])
print(result1[0:10])
df = pd.DataFrame({'file_name': sub['file_name'], 'label': result1})
now = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
fname="submit_"+now+r".csv"
df.to_csv(fname, index=False)

