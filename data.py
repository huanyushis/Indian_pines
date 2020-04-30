import scipy.io as scio
import numpy as np

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
# 挑选分类
# print(lens)
trains = []
labels = []
for i in lens:
    train =[indian_pines[index] for index in indexs[i[0]][::5]]
    trains.extend(train)
    labels.extend([i[0]]*len(train))
trains=np.array(trains)
labels=np.array(labels)

print(trains.shape)
print(labels.shape)