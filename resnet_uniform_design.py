import random
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Input, Activation,\
    BatchNormalization, GlobalAveragePooling2D, Add, multiply, Concatenate, GlobalMaxPooling2D, DepthwiseConv2D, SeparableConv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical
import keras
import sys
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pyunidoe as pydoe
import uniform_gen

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# datagen = ImageDataGenerator(
#     horizontal_flip=True, # 加入水平翻轉
#     brightness_range=[0.2, 1.0],  # 隨機調整亮度
#     shear_range=0.2,  # 剪切變換
#     fill_mode='nearest',
# )

start = time.time()
sys.setrecursionlimit(10000)

def filter_classes(x, y, class1, class2):
    idx = np.where((y == class1) | (y == class2))[0]
    x, y = x[idx], y[idx]
    y = np.where(y == class1, 0, 1)  # 將類別標籤轉換為 0 和 1
    return x, y

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# x_train, y_train = filter_classes(x_train, y_train, 3, 5)  # 例如，選擇貓（3）和狗（5）
# x_test, y_test = filter_classes(x_test, y_test, 3, 5)
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.166666, random_state=42)
print(x_train.shape)
print(x_test.shape)
x_train = x_train / 255.0
x_test = x_test / 255.0
# x_train = x_train.reshape(-1, 28, 28, 1)  # 這邊-1是代表不確定這邊的通道數是多少
# x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.reshape(-1, 32, 32, 3)  # 這邊-1是代表不確定這邊的通道數是多少
x_test = x_test.reshape(-1, 32, 32, 3)
datagen.fit(x_train)
datagen.fit(x_test)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


def conv (inputs, filters, kernel_size, activation, strides):
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same')(inputs)
    if activation:
        x = Activation(activation)(x)
    return x

def cbam(inputs, ratio = 8):
    cbaml = channel(inputs, ratio)
    cbaml = spatials(cbaml)

    return cbaml

def channel(inputs, ratio):
    channels = inputs.shape[-1]
    # print('shape', inputs.shape)
    shared_layer_1 = Dense(channels // ratio,
                           activation = 'relu',
                           kernel_initializer = 'he_normal', #權重矩陣初始化
                           use_bias = True, # 是否使用偏置向量(bias)
                           bias_initializer = 'zeros') # bias之initializer
    shared_layer_2 = Dense(channels,
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')
    x = gavgpool(inputs)
    x = shared_layer_1(x)
    x = shared_layer_2(x)

    x2 = gmaxpool(inputs)
    x2 = shared_layer_1(x2)
    x2 = shared_layer_2(x2)

    out_put = Add()([x, x2])
    out_put = Activation('sigmoid')(out_put)

    return multiply([inputs, out_put])



def spatials(inputs):

    class ReduceMean(tf.keras.layers.Layer):
        def call(self, x):
            return tf.reduce_mean(x, axis = -1, keepdims = True)

    class ReduceMax(tf.keras.layers.Layer):
        def call(self, x):
            return tf.reduce_max(x, axis = -1, keepdims = True)

    x1 = ReduceMean()(inputs)
    x2 = ReduceMax()(inputs)

    out_put = Concatenate(axis = -1)([x1, x2])
    # out_put = SeparableConv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation = 'sigmoid')(out_put)

    out_put = conv(out_put, 1, (3, 3), 'sigmoid', (1, 1))
    # out_put = BatchNormalization()(out_put)

    return multiply([inputs, out_put])

def gmaxpool(inputs):
    return GlobalMaxPooling2D()(inputs)

def gavgpool(inputs):
    return GlobalAveragePooling2D()(inputs)

def maxpool(inputs):
    return MaxPooling2D(strides = 2, pool_size = (3, 3), padding = 'same')(inputs)

def avgpool(inputs):
    return AveragePooling2D(strides = 2, pool_size = (3, 3), padding = 'same')(inputs)

def resnetblock(inputs, filters, filters2, activation, strides, kernel1, kernel2):
    # print(inputs.shape)
    x = SeparableConv2D(filters = filters, kernel_size = kernel1, strides = (1, 1), padding = 'same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # print(x.shape)
    x = SeparableConv2D(filters = filters2, kernel_size = kernel2, strides = (1, 1), padding = 'same')(x)
    x = BatchNormalization()(x)
    cbam_in = cbam(x)
    # x = Add()([x, cbam_in])

    # Adjust the shortcut path if needed
    if inputs.shape[-1] != filters2 or strides != (1, 1):
        # print('int', inputs.shape)
        res = Conv2D(filters = filters2, kernel_size = (1, 1), strides = strides, padding = 'same')(inputs)
        # res = conv(inputs, filters2, (1, 1), activation, strides)
        res = BatchNormalization()(res)
        # res = Activation(activation)(res)
    else:
        res = inputs
    x = Add()([cbam_in, res])
    x = Activation(activation)(x)
    return x

def resnet18(inputs, filter, filter2, kernel, kernel2, strides):
    # x = conv(inputs, filter, kernel, None, (2, 2))
    # x = BatchNormalization()(x)
    # x = conv(inputs, 64, (3, 3), None, (1, 1))
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = resnetblock(inputs, filter, filter2, 'relu', strides, kernel, kernel2)
    return x

def fitness(arr):
    # test_accuracy = np.random.random()
    arr = [sublist for sublist in arr if sublist[0] != 'N']

    print("arr: ", arr)
    inputs = Input(shape=(32, 32, 3), dtype="float32")
    feature = 32
    x = conv(inputs, 64, (7, 7), None, (2, 2))
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = maxpool(x)

    # for i in range(len(arr)):
    #     if (arr[i][0] == "R"):
    #         for j in range(1, 5):
    #             if(arr[i][j] == 0 or arr[i][j] < 0):
    #                 print('<=0', arr)
    #             elif(arr[i][j] > 256):
    #                 print('>', arr)

    for i in range(len(arr)):
        print(f'arrs{i}', arr[i])
        if(arr[i][0] == "R"):
            filters1 = arr[i][1]
            kernels1 = arr[i][2]
            filters2 = arr[i][3]
            kernels2 = arr[i][4]

            # if(arr[i][1] > arr[i][2]):
            #     filters = np.random.randint(arr[i][2], arr[i][1] + 1)
            # else:
            #     filters = np.random.randint(arr[i][1], arr[i][2] + 1)
            # if (arr[i][3] > arr[i][4]):
            #     kernel = np.random.randint(arr[i][4], arr[i][3] + 1)
            # else:
            #     kernel = np.random.randint(arr[i][3], arr[i][4] + 1)

            x = resnet18(x, filters1, filters2, (kernels1, kernels1), (kernels2, kernels2), (1, 1))
            # print("res")
            # print(x.shape)
        elif(arr[i][0] == "AP"):
            x = avgpool(x)
            # print("avg")
        elif(arr[i][0] == "MP"):
            x = maxpool(x)
            # print("max")

    print("arr: ", arr)
    x = GlobalAveragePooling2D()(x)
    # print('glo')
    outputs = Dense(10, activation='softmax')(x)
    # print('dense')

    model = Model(inputs=inputs, outputs=outputs)
    # print('model')
    optimizer = Adam(learning_rate = 0.001)
    # print('optimize')
    model.compile(optimizer = optimizer, loss = 'mse', metrics = ['acc'])
    # print('compile')
    model.fit(x_train, y_train,
          epochs=1,
          batch_size=20,
          validation_split=0.2)
    # print('fit')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("acc: ", test_accuracy)
    return test_accuracy

list1 = []
acc_list = []
cnt = 0
for t in range(1):
    R0 = ['R', 1, 2, 3, 4]
    AP = ['AP']
    MP = ['MP']
    for s in range(3, 8):
        stat = uniform_gen.ud(10, s)
        uniform = stat
        for j in range(0, s):
            cnt += 1
            table = uniform[j, :]
            rcnt = 0
            for i in table:
                if i % 3 == 1:
                    rcnt += 1
            list2 = []
            level = (rcnt + 1) * 2
            stat2 = uniform_gen.ud(level, 2)
            uniform2 = stat2
            # print(uniform2)
            col = 0
            row = 0
            R = copy.deepcopy(R0)
            R[1] = round((uniform2[row, col] / level) * 253) + 3
            R[2] = round((uniform2[row, col + 1] / level) * 4) + 3
            R[3] = round((uniform2[row + 1, col] / level) * 253) + 3
            R[4] = round((uniform2[row + 1, col + 1] / level) * 4) + 3
            list2.append(R)
            row = row + 2
            for i in table:
                # print('i', i)
                if i % 3 == 1:
                    R = copy.deepcopy(R0)
                    R[1] = round((uniform2[row, col] / level) * 253) + 3
                    R[2] = round((uniform2[row, col + 1] / level) * 4) + 3
                    R[3] = round((uniform2[row + 1, col] / level) * 253) + 3
                    R[4] = round((uniform2[row + 1, col + 1] / level) * 4) + 3
                    list2.append(R)
                    row = row + 2
                elif i % 3 == 2:
                    list2.append(AP)
                elif i % 3 == 0:
                    list2.append(MP)
            print(list2)
            fit = fitness(list2)
            acc_list.append(fit)
        print('step finish')
# print(cnt)
print(acc_list)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, cnt + 1), acc_list, color='red', marker='o', label='Test Accuracy')
# plt.xlabel('Experiment Index')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy over Experiments')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# plt.close()