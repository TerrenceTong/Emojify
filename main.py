import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


from flask import Flask, jsonify, render_template, request

all_tweets = pd.read_csv('./emojify/data/emojis_homemade1.csv')
tweets = all_tweets.groupby('emoji').filter(lambda c:len(c) > 1000)

train_tweets = tweets[0:100000]
x_train = train_tweets["text"]
tokenizer = Tokenizer(5000)
tokenizer.fit_on_texts (x_train)

'''从.h5文件加载模型'''
#model_test = load_model('./emojify/model/linear_classifier.h5')
#model_test.predict(tokenizer.texts_to_matrix("this is a test"))
model_1 = load_model('./emojify/model/emoji_linear.h5')
model_1.predict(tokenizer.texts_to_matrix("this is a test"))
model_2 = load_model('./emojify/emoji_hidden1.h5')
#model_3 = load_model('./emojify/emoji_hidden2.h5')
model_4 = load_model('./emojify/emoji_hidden2_dropout_L2.h5')
model_5 = load_model('./emojify/emoji_embeding1.h5')
model_6 = load_model('./emojify/emoji_embedding2_temp.h5')
model_7 = load_model('./emojify/emoji_lstm.h5')

# webapp
app = Flask(__name__)


@app.route('/api/model', methods=['POST'])
def model():
    x = request.json #这是从前端拿到的数据，类型为str
    #print(type(x))
    #print(x)
    x_input = tokenizer.texts_to_matrix(x)
    #print(x_input.shape)

    
    #print(x_input)
    output1 = model_1.predict(x_input)[0]
    output2 = model_2.predict(x_input)[0]
    #output3 = model_3.predict(x_input)[0]
    output4 = model_4.predict(x_input)[0]
    output5 = model_5.predict(x_input)[0]
    output6 = model_6.predict(x_input)[0]
    output7 = model_7.predict(x_input)[0]
    #output1 = model_test.predict_classes(x_input)
    #print(output1.shape)
    #print(output1)
    
    """ output1 = [0.001,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output2 = [0.002,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output3 = [0.003,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output4 = [0.004,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output5 = [0.005,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output6 = [0.006,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output7 = [0.007,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005] """
    """ results=[output1, output2, output3, output4, output5, output6, output7] """
    return jsonify(results=[output1, output2, output4, output5, output6, output7])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
