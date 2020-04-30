import scipy.io as scio
import numpy as np
import random
from sklearn.model_selection import train_test_split

# 145*145*200的数据集
indian_pines = scio.loadmat("databases/Indian_pines_corrected.mat")['indian_pines_corrected'].reshape(-1, 200)
# 145*145的标签
indian_pines_gt = scio.loadmat('databases/Indian_pines_gt.mat')['indian_pines_gt'].reshape(indian_pines.shape[0])

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
    indexs[j].append(i)
# 查看标签个数
lens = []
for key in indexs.keys():
    les = len(indexs[key])
    # print("%d:%d" % (key, les))
    lens.append((key, les))
# print(lens)
# 挑选分类
# print(lens)
x_trains = []
x_tests = []
y_trains = []
y_tests = []
for i in lens:
    X = [indian_pines_gt[index] for index in indexs[i[0]]]
    train, test ,y_train,y_test= train_test_split(X, [i[0]]*i[1], test_size=0.2)
    x_trains.extend(train)
    x_tests.extend(test)
    y_trains.extend(y_train)
    y_tests.extend(y_test)

x_trains = np.array(x_trains).astype(np.float64)
x_tests = np.array(x_tests).astype(np.float64)
y_trains = np.array(y_trains).astype(np.float64)
y_tests = np.array(y_tests).astype(np.float64)

print(x_trains.shape)
print(y_trains.shape)
print(x_tests.shape)
print(y_tests.shape)
# print(trains.dtype)
# print(labels.dtype)
