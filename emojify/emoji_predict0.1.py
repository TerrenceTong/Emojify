# coding: utf-8

# # Emoji 表情生成器

# ## 导入使用的库
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


# ## 对数据集进行统计分析
all_tweets = pd.read_csv('data/emojis_homemade.csv')
all_tweets['emoji'].value_counts()

all_tweets.shape

all_tweets.head

tweets = all_tweets.groupby('emoji').filter(lambda c:len(c) > 1000)
tweets['emoji'].value_counts()


chars = list(sorted(set(chain(*tweets['text']))))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
max_sequence_len = max(len(x) for x in tweets['text'])

emojis = list(sorted(set(tweets['emoji'])))
emoji_to_idx = {em: idx for idx, em in enumerate(emojis)}
emojis[:10]

#train_tweets, test_tweets = train_test_split(tweets, test_size=0.1)


# ## 数据集预处理
# 将数据集划分为train集，dev集，test集
# 对文本内容进行分词/one-hot编码
# 对emoji表情进行one-hot编码

#初始化train，dev，test集
train_tweets = tweets[0:100000]
dev_tweets = tweets[100000:110000]
test_tweets = tweets[110000:120000]
x_train = train_tweets["text"]
y_train = np.asarray(train_tweets["emoji"])
x_dev = dev_tweets["text"]
y_dev = np.asarray(dev_tweets["emoji"])
x_test = test_tweets["text"]
y_test = np.asarray(test_tweets["emoji"])
#将所有emoji数组拼接
all_emojis = np.concatenate((y_train, y_dev, y_test), axis=0)
print(all_emojis)
print(all_emojis.shape)
#对emoji表情进行编号
emoji_to_idx = {em: idx for idx, em in enumerate(emojis)}
#初始化all_emojis_idx矩阵
all_emojis_idx = np.zeros(all_emojis.shape[0])
for i in range (all_emojis.shape[0]):
    all_emojis_idx[i] = emoji_to_idx[all_emojis[i]]
print(all_emojis_idx)
print(all_emojis_idx.shape)
#对all_emojis_idx矩阵进行one-hot编码
all_emojis_one_hot = to_categorical (all_emojis_idx)
print(all_emojis_one_hot)
print(all_emojis_one_hot.shape)
#对进行one-hot编码的数据重新划分train，dev，test
y_train_idx = all_emojis_one_hot[0:100000,:]
y_dev_idx = all_emojis_one_hot[100000:110000,:]
y_test_idx = all_emojis_one_hot[110000:120000,:]

#对x_train进行分词
num_words = 5000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts (x_train)

#将分词结果按照次数排序
#word_count = sorted(list(tokenizer.word_counts.items()), key=lambda x: x[1], reverse=True)
#print(word_count)

#根据分词器的结果，对train，dev，test数据集进行one-hot编码
x_train_one_hot = tokenizer.texts_to_matrix(x_train, mode='binary')
x_dev_one_hot = tokenizer.texts_to_matrix(x_dev, mode='binary')
x_test_one_hot = tokenizer.texts_to_matrix(x_test, mode='binary')

#输出train,dev,test的shape进行检验
print(x_train_one_hot.shape)
print(y_train_idx.shape)
print(x_dev_one_hot.shape)
print(y_dev_idx.shape)
print(x_test_one_hot.shape)
print(y_test_idx.shape)

# ## 线性分类器
# * 输入层
# * 输出层：全连接层
# 

#设置模型输出维度
output_size = y_dev_idx.shape[1]
#采用序贯模型
model = Sequential()
#加入全连接层
model.add(Dense(output_size, activation='softmax', input_shape=(num_words,))) 
#对模型进行编译
model.compile(optimizer=optimizers.RMSprop(lr = 0.005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#对模型进行训练
history = model.fit (x_train_one_hot, y_train_idx,
                    epochs = 2,
                    batch_size = 512,
                    validation_data = (x_dev_one_hot, y_dev_idx),
                    verbose=2)

#2018.12.27,TongQi,保存模型
model.save('./model/linear_classifier.h5')
print('stored!')

import matplotlib.pyplot as plt

#定义绘制模型准确率的曲线
def plot_train_history(history): 
    #初始化train，validation集合的准确率和损失值
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #初始化周期数
    epochs = range(1, len(acc) + 1)
    #绘制准确率曲线
    plt.plot (epochs, acc, 'bo', label='Training acc')
    plt.plot (epochs, val_acc, 'b', label='Validation acc')
    plt.title ('Training and validation accuracy')
    plt.legend()
    plt.figure()
    #绘制loss曲线
    plt.plot(epochs, loss, 'bo', label = 'Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()



print ("best validation accuracy: ", max(history.history['val_acc']))
plot_train_history(history)


# ## 一层隐含层的神经网络
# * 输入层
# * 隐含层：全连接层
# * 输出层：全连接层


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
              metrics=['accuracy'])
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
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))


# ## 两层隐含层的神经网络
# * 输入层
# * 隐含层1：全连接层
# * 隐含层2：全连接层
# * 输出层：全连接层

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
              metrics=['accuracy'])
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
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))


# ## 两层隐含层的神经网络：dropout和正则化
# * 输入层
# * 隐含层1：全连接层，dropout，L2正则化
# * 隐含层2：全连接层，dropout，L2正则化
# * 输出层：全连接层

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
              metrics=['accuracy'])
#对模型进行训练
history = model.fit (x_train_one_hot, y_train_idx,
                    epochs = 20,
                    batch_size = 512,
                    verbose=1,
                    validation_data = (x_dev_one_hot, y_dev_idx))
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))


# ## 加入embedding层的简单神经网络
# * 输入层
# * 隐含层1：embedding层
# * 隐含层2:   flatten层
# * 输出层：全连接层


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
                  metrics=['accuracy'])
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
checkpoint = keras.callbacks.ModelCheckpoint(filepath='emoji_embedding.h5',
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
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))


# ## 加入embedding层的深层神经网络
# * 输入层
# * 隐含层1:  embedding层
# * 隐含层2:  flatten层
# * 隐含层3:  全连接层，dropout，L2正则化
# * 隐含层4:  全连接层，dropout，L2正则化
# * 输出层：全连接层


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
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))


# ## 模型调参


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
#绘制曲线
plot_train_history(history)
print ("best validation accuracy: ", max(history.history['val_acc']))


# ## LSTM模型
# * 输入层
# * 隐含层1：embedding层
# * 隐含层2:  LSTM层
# * 隐含层3:  全连接层，dropout，正则化
# * 隐含层4:  全连接层，dropout，正则化
# * 输出层：全连接层


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