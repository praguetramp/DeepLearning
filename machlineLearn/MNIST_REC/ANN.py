# 人工神经网络应用于MNIST数据集

import pandas as pd
from tensorflow.python.keras import layers, models

input_data = pd.read_csv("train.csv")

y = input_data['label']
# 删除数据中的     label      列删除(默认为行删除)      直接在原数据内存替换(False为新的内存)
input_data.drop('label', axis=1, inplace=True)
X = input_data
# 根据y中0-9个数字进行分类并简历标签独热编码，所在类别为1
y = pd.get_dummies(y)

# 创建模型，保存输入层、输出层、隐藏层
# Sequential()方法是一个容器，描述了神经网络的网络结构
classifier = models.Sequential()

# dense：全连接层  cov2d： 卷积层
'''
    Dense参数：
                 units:设置该层节点数，也可以看成对下一层的输入。
                 activation：激活函数，在这一层输出的时候是否需要激活函数
                 use_bias：偏置，默认带有偏置。
                 kernel_initializer:权重初始化方法
                 bias_initializer:偏置值初始化方法
                 kernel_regularizer:权重规范化函数
                 bias_regularizer:偏置值规范化方法
                 activity_regularizer:输出的规范化方法
                 kernel_constraint:权重变化限制函数
                 bias_constraint:偏置值变化限制函数
'''
classifier.add(layers.Dense(units=600, activation='relu', kernel_initializer='uniform', input_dim=784))
classifier.add(layers.Dense(units=400, activation='relu', kernel_initializer='uniform', input_dim=784))
classifier.add(layers.Dense(units=200, activation='relu', kernel_initializer='uniform', input_dim=784))
classifier.add(layers.Dense(units=10, activation='sigmoid', kernel_initializer='uniform', input_dim=784))
# 随机梯度下降(stochastic gradient descent)算法最小化损失，>>>反向传播<<<
classifier.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
# 指定 批次大小和循环次数  进行训练
classifier.fit(X, y, batch_size=10, epochs=10)

# 在训练过的神经网络模型中进行测试输出
test_data = pd.read_csv('test.csv')
y_pred = classifier.predict(test_data)
