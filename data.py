import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

tf.keras.backend.set_floatx('float64')


def split_data(datas, lables, p_train, p_test, p_val=0):
    lens = datas.shape[0]
    sum = (p_train + p_test + p_val)
    val_x, val_y = None, None
    if p_val:
        datas, val_x, lables, val_y = train_test_split(datas, lables, test_size=p_val / (sum))
    train_x, test_x, train_y, test_y = train_test_split(datas, lables, test_size=p_test / (sum - p_val))
    return train_x, train_y, test_x, test_y, val_x, val_y


# 从.mat文件加载Indian_pines数据集
def load_data():
    Indian_pines_corrected = loadmat('databases/Indian_pines_corrected.mat')['indian_pines_corrected']
    Indian_pines_corrected=Indian_pines_corrected/Indian_pines_corrected.max()
    Indian_pines_corrected.astype('float64')
    Indian_pines_gt = loadmat('databases/Indian_pines_gt.mat')['indian_pines_gt'].astype('float64')
    return Indian_pines_corrected, Indian_pines_gt


def get_data():
    Indian_pines, Indian_pines_gt = load_data()
    # 查看数据的结构
    # Indian_pines.shape is (145, 145, 200),Indian_pines_gt.shape is (145, 145)
    # print(f"Indian_pines.shape is {Indian_pines.shape},Indian_pines_gt.shape is {Indian_pines_gt.shape}")
    # 转化为(145*145,200) (145*145,)
    Indian_pines = Indian_pines.reshape(-1, 200)
    Indian_pines_gt = Indian_pines_gt.reshape(-1, 1)
    # print(f"number of type is {np.unique(Indian_pines_gt).shape[0]}")
    # Indian_pines.shape is (21025, 200),Indian_pines_gt.shape is (21025, 1)
    # print(f"Indian_pines.shape is {Indian_pines.shape},Indian_pines_gt.shape is {Indian_pines_gt.shape}")

    # Indian_pines_gt = to_categorical(Indian_pines_gt, np.unique(Indian_pines_gt).shape[0])
    # 划分数据集为训练集、测试集、验证集 (6:3:1)
    return split_data(Indian_pines, Indian_pines_gt, 6, 2, 2)


train_x, train_y, test_x, test_y, val_x, val_y = get_data()


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(1024, activation='relu')
        self.d3 = Dense(1024, activation='relu')
        self.d4 = Dense(17, activation='sigmoid')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


model = MyModel()
model.build(input_shape=(None, 200))
# print(model.summary())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 10000
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    train_step(train_x, train_y)
    test_step(test_x, test_y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
    if not epoch:
        best_val = test_accuracy.result()

    if test_accuracy.result() > best_val:
        print(f"acc提高了，保存模型")
    else:
        print("acc没有提高")
