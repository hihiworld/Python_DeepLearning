from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# 向量化输入数据
def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(sequences, dimension=46):
    results = np.zeros((len(sequences), dimension))
    for sequence, i in enumerate(sequences):
        results[sequence][i] = 1
    return results

# one-hot编码
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)

# one-hot编码Keras实现 
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# 整数型编码
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# 创建模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 测试集验证集划分
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

# y_val = one_hot_train_labels[:1000]
# partial_y_train = one_hot_train_labels[1000:]
y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train,batch_size=512, epochs=20, validation_data=(x_val, y_val))

history_dict = history.history

# 绘制损失值

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = [i for i in range(1, len(loss)+1)]

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Val Training Loss')
plt.title('Loss and Epochs Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
