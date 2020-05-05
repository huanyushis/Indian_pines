import os
import random
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, LeakyReLU
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

tf.keras.backend.set_floatx('float64')


# def split_data(datas, lables, p_train, p_test, p_val=0):
#     lens = datas.shape[0]
#     sum = (p_train + p_test + p_val)
#     val_x, val_y = None, None
#     if p_val:
#         datas, val_x, lables, val_y = train_test_split(datas, lables, test_size=p_val / (sum))
#     train_x, test_x, train_y, test_y = train_test_split(datas, lables, test_size=p_test / (sum - p_val))
#     return train_x, train_y, test_x, test_y, val_x, val_y


# 从.mat文件加载Indian_pines数据集
def load_data():
    Indian_pines_corrected = loadmat('databases/Indian_pines_corrected.mat')['indian_pines_corrected']
    Indian_pines_corrected = Indian_pines_corrected / Indian_pines_corrected.max()
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

    index = list(range(Indian_pines.shape[0]))
    random.shuffle(index)
    Indian_pines = Indian_pines[index]
    Indian_pines_gt = Indian_pines_gt[index]
    # 划分数据集为训练集、测试集、验证集 (6:3:1)
    return Indian_pines[:int(Indian_pines.shape[0] * 0.6)], Indian_pines_gt[
                                                            :int(Indian_pines.shape[0] * 0.6)], Indian_pines[int(
        Indian_pines.shape[0] * 0.6):int(Indian_pines.shape[0] * 0.9)], Indian_pines_gt[
                                                                        int(Indian_pines.shape[0] * 0.6):int(
                                                                            Indian_pines.shape[0] * 0.9)], Indian_pines[
                                                                                                           int(
                                                                                                               Indian_pines.shape[
                                                                                                                   0] * 0.9):], Indian_pines_gt[
                                                                                                                                int(
                                                                                                                                    Indian_pines.shape[
                                                                                                                                        0] * 0.9):]


get_data()
train_x, train_y, test_x, test_y, val_x, val_y = get_data()


# print(train_x.shape)
# print(train_y.shape)

# 构建模型
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(1024)
        self.d2 = Dense(1024)
        self.d3 = Dense(1024)
        self.d4 = Dense(1024)
        self.d5 = Dense(17, activation="softmax")
        self.LeakyReLU = LeakyReLU()
        self.drop = Dropout(0.25)

    def call(self, x):
        x = self.d1(x)
        x = self.LeakyReLU(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.LeakyReLU(x)
        x = self.drop(x)
        x = self.d3(x)
        x = self.LeakyReLU(x)
        x = self.drop(x)
        x = self.d4(x)
        x = self.LeakyReLU(x)
        x = self.drop(x)
        output = self.d5(x)
        return output


model = MyModel()
model.build(input_shape=(None, 200))
print(model.summary())
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='test_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = tf.reduce_sum(loss_object(labels, predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    # print(gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def val_step(images, labels):
    predictions = model(images)
    t_loss = tf.reduce_sum(loss_object(labels, predictions))

    val_loss(t_loss)
    val_accuracy(labels, predictions)


EPOCHS = 10000
for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    train_step(train_x, train_y)
    val_step(val_x, val_y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          val_loss.result(),
                          val_accuracy.result() * 100))
    # if not epoch:
    #     best_val = val_accuracy.result()
    #     print("第一次保存模型")
    #     tf.saved_model.save(model, "save")
    # if val_accuracy.result() > best_val:
    #     print(f"val_acc提升，保存模型,acc:{val_accuracy}")
    #     tf.saved_model.save(model, "save")
    # else:
    #     print("val_acc没有提高")
    # print("test_acc:",categorical_crossentropy(model(test_x),test_y))
