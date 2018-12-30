import random
import twitter
import emoji
# import itertools
import pandas as pd
from itertools import chain
from keras.preprocessing.sequence import pad_sequences
import numpy as np
#from sklearn.model_selection import train_test_split
from keras import Sequential, optimizers, regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.utils import to_categorical
import keras.callbacks
from keras.backend import clear_session
#import json

import os
# import nb_utils
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Embedding, GlobalMaxPooling1D#, Merge 
from keras.models import Model
from keras.layers.merge import Concatenate, Average

# from gensim.models import Word2Vec

'''对数据集进行统计分析'''
#读入数据集
all_tweets = pd.read_csv('data/emojis_homemade1.csv')
#对数据集中的表情进行统计
all_tweets['emoji'].value_counts()

#输出数据集的shape
all_tweets.shape

#输出数据集的前10条数据
all_tweets.head(10)

#保留数据集中出现次数超过1000次的数据
tweets = all_tweets.groupby('emoji').filter(lambda c:len(c) > 1000)
#对数据集中的数据进行统计
tweets['emoji'].value_counts()

#chars中保存了tweets中的text
chars = list(sorted(set(chain(*tweets['text']))))
#将chars转换为idx
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
#求tweets中最长的文本长度
max_sequence_len = max(len(x) for x in tweets['text'])
#emojis保存了tweets中的表情
emojis = list(sorted(set(tweets['emoji'])))
#将emoji转换为idx
emoji_to_idx = {em: idx for idx, em in enumerate(emojis)}
#emojis[:10]

'''数据集预处理
将数据集划分为train集，dev集，test集
对文本内容进行分词/one-hot编码
对emoji表情进行one-hot编码'''
#初始化train集
train_tweets = tweets[0:680000]
#初始化dev集
dev_tweets = tweets[680000:700000]
#初始化test集
test_tweets = tweets[700000:720000]
#x_train初始化为train中的文本信息
x_train = train_tweets["text"]
#y_train初始化为train中的表情
y_train = np.asarray(train_tweets["emoji"])
#x_dev初始化为dev中的文本信息
x_dev = dev_tweets["text"]
#y_dev初始化为dev中的表情
y_dev = np.asarray(dev_tweets["emoji"])
#x_test初始化为test中的文本信息
x_test = test_tweets["text"]
#y_test初始化为test中的表情
y_test = np.asarray(test_tweets["emoji"])
#将所有emoji数组拼接
all_emojis = np.concatenate((y_train, y_dev, y_test), axis=0)
print(all_emojis)
print(all_emojis.shape)
#对emoji表情进行编号
emoji_to_idx = {em: idx for idx, em in enumerate(emojis)}
#初始化all_emojis_idx矩阵
all_emojis_idx = np.zeros(all_emojis.shape[0])
#对emoji进行编号
for i in range (all_emojis.shape[0]):
    all_emojis_idx[i] = emoji_to_idx[all_emojis[i]]
print(all_emojis_idx)
print(all_emojis_idx.shape)
#对all_emojis_idx矩阵进行one-hot编码
all_emojis_one_hot = to_categorical (all_emojis_idx)
print(all_emojis_one_hot)
print(all_emojis_one_hot.shape)
#对进行one-hot编码的数据重新划分train
y_train_idx = all_emojis_one_hot[0:680000,:]
#对进行one-hot编码的数据重新划分dev
y_dev_idx = all_emojis_one_hot[680000:700000,:]
#对进行one-hot编码的数据重新划分test
y_test_idx = all_emojis_one_hot[700000:720000,:]


#设置分词的数目
num_words = 5000
#初始化分词器
tokenizer = Tokenizer(num_words=num_words)
#对x_train进行分词
tokenizer.fit_on_texts (x_train)

#将分词结果按照次数排序
#word_count = sorted(list(tokenizer.word_counts.items()), key=lambda x: x[1], reverse=True)
#print(word_count)

#根据分词器的结果，对train数据集进行one-hot编码
x_train_one_hot = tokenizer.texts_to_matrix(x_train, mode='binary')
#根据分词器的结果，对dev数据集进行one-hot编码
x_dev_one_hot = tokenizer.texts_to_matrix(x_dev, mode='binary')
#根据分词器的结果，对tes数据集进行one-hot编码
x_test_one_hot = tokenizer.texts_to_matrix(x_test, mode='binary')

#输出train,dev,test的shape进行检验
print(x_train_one_hot.shape)
print(y_train_idx.shape)
print(x_dev_one_hot.shape)
print(y_dev_idx.shape)
print(x_test_one_hot.shape)
print(y_test_idx.shape)


import matplotlib.pyplot as plt
'''定义训练时需要用到的函数'''
#定义绘制模型准确率的曲线
def plot_train_history(history): 
    #初始化train集合的准确率
    acc = history.history['top_k_categorical_accuracy']
    #初始化validation集合的准确率
    val_acc = history.history['val_top_k_categorical_accuracy']
    #初始化train集合的损失值
    loss = history.history['loss']
    #初始化validation集合的准确率
    val_loss = history.history['val_loss']
    #初始化周期数
    epochs = range(1, len(acc) + 1)
    #绘制Training acc曲线
    plt.plot (epochs, acc, 'bo', label='Training acc')
    #绘制Validation acc曲线
    plt.plot (epochs, val_acc, 'b', label='Validation acc')
    #绘制曲线标题
    plt.title ('Training and validation accuracy')
    #绘制曲线标题
    plt.legend()
    #绘制figure窗口
    plt.figure()
    #绘制Training loss曲线
    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    #绘制Validation loss曲线
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    #绘制曲线标题
    plt.title('Training and validation loss')
    #绘制曲线标题
    plt.legend()
    #展示图片
    plt.show()


'''线性分类器
输入层
输出层：全连接层'''
#设置模型输出维度
output_size = y_dev_idx.shape[1]
#采用序贯模型
model = Sequential()
#加入全连接层
model.add(Dense(output_size, activation='softmax', input_shape=(num_words,))) 
#对模型进行编译
model.compile(optimizer=optimizers.RMSprop(lr = 0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy','top_k_categorical_accuracy'])
#对模型进行训练
history = model.fit (x_train_one_hot, y_train_idx,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_dev_one_hot, y_dev_idx),
                    verbose=2)

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_linear.h5")


#在测试集上进行测试
score = model.evaluate(x_test_one_hot, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])


model.predict_classes(x_test_one_hot)

'''一层隐含层的神经网络
输入层
隐含层：全连接层
输出层：全连接层'''
#设置模型输出维度
n_output_classes = y_dev_idx.shape[1]
#采用序贯模型
model = Sequential()
#加入全连接层1
model.add(Dense(128, activation='relu', input_shape=(num_words,)))
#加入全连接层2
model.add(Dense(n_output_classes, activation = 'softmax'))
#对模型进行编译
model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy','top_k_categorical_accuracy'])
#加入early stopping防止过拟合
early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.01,
                              patience=10,
                              verbose=1, mode='auto')
#对模型进行训练
history = model.fit (x_train_one_hot, y_train_idx,
                    epochs = 10,
                    batch_size = 512,
                    verbose=1,
                    validation_data = (x_dev_one_hot, y_dev_idx),
                    callbacks = [early])


#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_hidden1.h5")

#在测试集上进行测试
score = model.evaluate(x_test_one_hot, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])


'''两层隐含层的神经网络
输入层
隐含层1：全连接层
隐含层2：全连接层
输出层：全连接层'''
#设置模型输出维度
n_output_classes = y_dev_idx.shape[1]
#采用序贯模型
model = Sequential()
#加入全连接层1
model.add(Dense(300, activation='relu', input_shape=(num_words,)))
#加入全连接层2
model.add(Dense(128, activation='relu', 
                kernel_regularizer = regularizers.l2(0.0025)))
#加入全连接层3
model.add(Dense(n_output_classes, activation = 'softmax'))
#对模型进行编译
model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy','top_k_categorical_accuracy'])
#加入early stopping防止过拟合
early = keras.callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0.01,
                              patience=5,
                              verbose=1, mode='auto')
#对模型进行训练
history = model.fit (x_train_one_hot, y_train_idx,
                    epochs = 20,
                    batch_size = 512,
                    verbose=1,
                    validation_data = (x_dev_one_hot, y_dev_idx),
                    callbacks=[early])

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_hidden2.h5")

#在测试集上进行测试
score = model.evaluate(x_test_one_hot, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])

'''两层隐含层的神经网络：dropout和正则化
输入层
隐含层1：全连接层，dropout，L2正则化
隐含层2：全连接层，dropout，L2正则化
输出层：全连接层'''
#设置模型输出维度
n_output_classes = y_dev_idx.shape[1]
#采用序贯模型
model = Sequential()
#加入全连接层1
model.add(Dense(300, activation='relu', input_shape=(num_words,)))
model.add(Dropout(0.4))
#加入全连接层2
model.add(Dense(128, activation='relu',
                kernel_regularizer = regularizers.l2(0.002)))
model.add(Dropout(0.4))
#加入全连接层3
model.add(Dense(n_output_classes, activation = 'softmax'))
#对模型进行编译
model.compile(optimizer = 'rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy','top_k_categorical_accuracy'])
#对模型进行训练
history = model.fit (x_train_one_hot, y_train_idx,
                    epochs = 20,
                    batch_size = 512,
                    verbose=1,
                    validation_data = (x_dev_one_hot, y_dev_idx))

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_hidden2.h5")

#在测试集上进行测试
score = model.evaluate(x_test_one_hot, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])


'''加入embedding层的简单神经网络
输入层
隐含层1：embedding层
隐含层2: flatten层
输出层：全连接层'''
#定义序列的最长长度为25
maxlen = 25
#对文本进行填充
def convert_to_sequences(tweet_text, tokenizer, maxlen=20):
    tweet_sequence = np.asarray(tokenizer.texts_to_sequences(tweet_text))
    padded = pad_sequences (tweet_sequence, maxlen=maxlen)
    return (padded)
#将train，dev，test集进行文本填充转换
x_train_sequences = convert_to_sequences(x_train, tokenizer, maxlen=maxlen)
x_dev_sequences = convert_to_sequences(x_dev, tokenizer, maxlen=maxlen)
x_test_sequences = convert_to_sequences(x_test, tokenizer, maxlen=maxlen)

#设置输出维度大小
n_output_classes = y_dev_idx.shape[1]

#定义embedding模型
def simple_embedding_model(num_words, n_output_classes, n_embedding_dims = 16, max_sequence_length = 20):
    #采用序贯模型
    model = Sequential()
    #加入embedding层
    model.add(Embedding(num_words, n_embedding_dims, input_length = max_sequence_length, name="embedding")) 
    #加入flatten层，把tensor转化为2维（samples，maxlen*8）
    model.add(Flatten()) 
    #加入全连接层
    model.add(Dense(n_output_classes, activation = 'softmax'))
    #对模型进行编译
    model.compile(optimizer = 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy','top_k_categorical_accuracy'])
    return (model)

clear_session()
#定义模型
model = simple_embedding_model(num_words,
                               n_output_classes, 
                               n_embedding_dims = 64, 
                               max_sequence_length = maxlen)    
#加入early stopping防止过拟合
early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.03,
                              patience=3,
                              verbose=1, mode='auto')
#加入断点续训
checkpoint = keras.callbacks.ModelCheckpoint(filepath='/model/emoji_embedding.h5',
                                             monitor='val_acc',
                                             save_best_only = True)
#对模型进行可视化
tensorboard = keras.callbacks.TensorBoard(log_dir='tensorboard_log',
                                          #write_grads=1,
                                          #histogram_freq=1,
                                          embeddings_freq=1,
                                          embeddings_data='embedding') 
#对模型进行训练
history = model.fit (x_train_sequences, y_train_idx,
                     validation_data = (x_dev_sequences, y_dev_idx),
                     epochs = 20,
                     batch_size = 512,
                     verbose=1,
                     callbacks=[early])

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_embe1.h5")

#在测试集上进行测试
score = model.evaluate(x_test_sequences, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])

model.predict_classes(x_test_sequences)


'''加入embedding层的深层神经网络
输入层
隐含层1: embedding层
隐含层2: flatten层
隐含层3: 全连接层，dropout，L2正则化
隐含层4: 全连接层，dropout，L2正则化
输出层：全连接层'''
#设置输出维度大小
n_output_classes = y_dev_idx.shape[1]
#定义deeper embedding模型
def deeper_embedding_model(num_words,
                           n_output_classes, 
                           n_embedding_dims = 16, 
                           max_sequence_length = 20, 
                           dense1_size = 16, 
                           dropout1_rate = 0.2,
                           dense2_size = 16,
                           dropout2_rate = 0.2,
                           lambd = 0.0):
    #采用序贯模型
    model = Sequential()
    #加入embedding层
    model.add(Embedding(num_words, n_embedding_dims, input_length = max_sequence_length, name="embedding")) 
    #加入flatten层
    model.add(Flatten()) 
    #加入全连接层
    model.add(Dense(dense1_size, activation='relu'))
    model.add(Dropout(dropout1_rate))
    #加入全连接层
    model.add(Dense(dense2_size, activation='relu',
                    kernel_regularizer = regularizers.l2(lambd)))
    model.add(Dropout(dropout2_rate))
    #加入全连接层
    model.add(Dense(n_output_classes, activation = 'softmax'))
    #对模型进行编译
    model.compile(optimizer = 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return (model)

'''模型调参'''
clear_session()
#定义模型
model = deeper_embedding_model(num_words,
                               n_output_classes, 
                               n_embedding_dims = 28, 
                               max_sequence_length = maxlen,
                               lambd = 0.0006)    
#加入early stopping防止过拟合
early = keras.callbacks.EarlyStopping(monitor='val_acc',
                              min_delta=0.01,
                              patience=5,
                              verbose=1, mode='auto')
#加入断点续训
checkpoint = keras.callbacks.ModelCheckpoint(filepath='emoji_embedding_v2.h5',
                                             monitor='val_acc',
                                             save_best_only = True)
#对模型进行训练
history = model.fit (x_train_sequences, y_train_idx,
                     validation_data = (x_dev_sequences, y_dev_idx),
                     epochs = 50,
                     batch_size = 512,
                     verbose=1,
                     callbacks=[early])

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_embedding2.h5")

clear_session()
#定义模型
model = deeper_embedding_model(num_words,
                               n_output_classes, 
                               n_embedding_dims = 128,          #196      # 128
                               max_sequence_length = maxlen,
                               dense1_size = 128,               # 160     # 128
                               dropout1_rate = 0.4,           # 0.35
                               dense2_size = 96,             # 128   # 96
                               dropout2_rate = 0.4,           # 0.35
                               lambd = 0.0025)                # 0.0015
#加入early stopping防止过拟合                           
early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.01,
                              patience=10,
                              verbose=1, mode='auto')
#对模型进行训练
history = model.fit (x_train_sequences, y_train_idx,
                     validation_data = (x_dev_sequences, y_dev_idx),
                     epochs = 50,
                     batch_size = 512,
                     verbose=1,
                     callbacks=[early])

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_embedding2_temp.h5")

#在测试集上进行测试
score = model.evaluate(x_test_sequences, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])

# LSTM模型
# 输入层
# 隐含层1：embedding层
# 隐含层2: LSTM层
# 隐含层3: 全连接层，dropout，正则化
# 隐含层4: 全连接层，dropout，正则化
# 输出层：全连接层

#定义LSTM模型
def simple_lstm_model(num_words,
                           n_output_classes, 
                           n_embedding_dims = 16, 
                           max_sequence_length = 20, 
                           dense1_size = 16, 
                           dropout1_rate = 0.2,
                           lambd = 0.0):
    #采用序贯模型
    model = Sequential()
    #加入embedding层
    model.add(Embedding(num_words, n_embedding_dims, input_length = max_sequence_length, name="embedding")) 
    #加入LSTM层
    model.add(LSTM(n_embedding_dims))
    # model.add(Flatten()) 
    #加入全连接层
    model.add(Dense(dense1_size, activation='relu'))
    model.add(Dropout(dropout1_rate))
    #加入全连接层
    model.add(Dense(n_output_classes, activation = 'softmax'))
    #对模型进行编译
    model.compile(optimizer = 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return (model)

clear_session()
n_output_classes = y_dev_idx.shape[1]
#定义模型
model = simple_lstm_model(num_words,
                          n_output_classes, 
                          n_embedding_dims = 90,          
                          max_sequence_length = maxlen,
                          dense1_size = 128,               
                          dropout1_rate = 0.2,           
                          lambd = 0.0025)                
#对模型进行可视化
tensorboard = keras.callbacks.TensorBoard(log_dir='tensorboard_log')
                                          #write_grads=1,
                                          #histogram_freq=1,
                                          #embeddings_freq=1,
                                          #embeddings_data='embedding') 
#加入early stopping防止过拟合  
early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.01,
                              patience=5,
                              verbose=1, mode='auto')
#对模型进行训练
history = model.fit (x_train_sequences, y_train_idx,
                     validation_data = (x_dev_sequences, y_dev_idx),
                     epochs = 50,
                     batch_size = 512,
                     verbose=1,
                     callbacks=[early, tensorboard])
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))

#输出最高的准确率
print ("best validation accuracy: ", max(history.history['val_top_k_categorical_accuracy']))
#绘制曲线
plot_train_history(history)
#保存模型
model.save("emoji_lstm.h5")

#在测试集上进行测试
score = model.evaluate(x_test_sequences, y_test_idx, verbose = 0)
#输出top5的准确率
print('Test top5 accuracy:', score[2])
