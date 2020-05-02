import scipy.io as scio
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorboard
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import sgd
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras import Model
# 145*145*200的数据集
indian_pines = scio.loadmat("databases/Indian_pines_corrected.mat")['indian_pines_corrected'].reshape(-1, 200)
maxs=indian_pines.max()
# 145*145的标签
indian_pines_gt = scio.loadmat('databases/Indian_pines_gt.mat')['indian_pines_gt'].reshape(indian_pines.shape[0])
indian_pines_gt-=1
# print(indian_pines.shape)
# print(indian_pines_gt.shape)
# 类别
indian_pines_labels = list(set(indian_pines_gt))
# print(indian_pines_labels)
# 类别总数
numType = len(indian_pines_labels)
# print(numType)

# 获取为某个值的索引
# print(indian_pines_gt[indian_pines_gt==2])
indexs = dict([(k, []) for k in indian_pines_labels])
# 获得每个标签所对应值的索引
for i, j in enumerate(indian_pines_gt):
    indexs[j].append(i)  # 查看标签个数
lens = []
for key in indexs.keys():
    if key!=255:
        les = len(indexs[key])
        # print("%d:%d" % (key, les))
        lens.append((key, les))
# print(lens)
# 挑选分类
x_trains_index = []
x_tests_index = []
x_vals_index = []
x_trains = []
x_tests = []
x_vals = []
y_trains = []
y_tests = []
y_vals = []
for i in lens:
    x_train, x_test, y_train, y_test = train_test_split(indexs[i[0]], [i[0]] * i[1], test_size=0.5)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.2)
    x_trains_index.extend(x_train)
    x_tests_index.extend(x_test)
    x_vals_index.extend(x_val)
    x_trains.extend([indian_pines[i, :] for i in x_train])
    x_tests.extend([indian_pines[i, :] for i in x_test])
    x_vals.extend([indian_pines[i, :] for i in x_val])
    y_trains.extend(y_train)
    y_tests.extend(y_test)
    y_vals.extend(y_val)
x_trains = np.array(x_trains).astype(np.float64)
x_tests = np.array(x_tests).astype(np.float64)
x_vals = np.array(x_vals).astype(np.float64)
y_trains = to_categorical(y_trains,num_classes=16)
y_tests = to_categorical(y_tests,num_classes=16)
y_vals = to_categorical(y_vals,num_classes=16)
x_trains /= maxs
x_tests /= maxs
x_vals /= maxs
print(x_trains.shape)
print(x_tests.shape)
print(x_vals.shape)
print(y_trains.shape)
print(y_tests.shape)
print(y_vals.shape)
# print(y_trains.shape)
# print(x_tests.shape)
# print(y_tests.shape)
# print(len(x_trains_index))
# print(len(x_tests_index))
inputs=Input(shape=(200,))
y=Dense(1024, activation='relu')(inputs)
y=Dropout(0.6)(y)
y=Dense(1024, activation='relu')(y)
y=Dropout(0.6)(y)
y=Dense(1024, activation='relu')(y)
y=Dropout(0.6)(y)
y=Dense(1024, activation='relu')(y)
y=Dropout(0.6)(y)
output=Dense(16, activation='relu')(y)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = Model(inputs=inputs, outputs=output)
print(model.summary())
model.compile(optimizer=adam,
              loss='mean_squared_error',
              metrics=['accuracy']
              )
checkpoint = ModelCheckpoint(filepath="ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_acc{acc:.3f}.h5",
                             monitor='acc',
                             verbose=1,
                             save_best_only='True',
                             mode='max',
                             period=1)
model.fit(x_trains, y_trains,
          validation_data=(x_vals, y_vals),
          callbacks = [checkpoint],
          verbose = 2,
          epochs=20000
)
model.save("model.h5")
score = model.evaluate(x_tests, y_tests, batch_size=1024)
print(score)
