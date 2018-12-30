import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


from flask import Flask, jsonify, render_template, request

""" x = tf.placeholder("float", [None, 784])
sess = tf.Session()

# restore trained data
with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/regression.ckpt")


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

 """

all_tweets = pd.read_csv('./emojify/data/emojis_homemade.csv')
tweets = all_tweets.groupby('emoji').filter(lambda c:len(c) > 1000)
train_tweets = tweets[0:100000]
x_train = train_tweets["text"]
tokenizer = Tokenizer(5000)
tokenizer.fit_on_texts (x_train)
model_linear_classifier = load_model("./emojify//model/linear_classifier.h5")

# webapp
app = Flask(__name__)


@app.route('/api/model', methods=['POST'])
def model():
    x = request.json
    print(type(x))
    print(x)
    """ input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784) """

    """ output1 = regression(input)
    output2 = convolutional(input) """
    output1 = [0.001,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output2 = [0.002,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output3 = [0.003,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output4 = [0.004,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output5 = [0.005,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output6 = [0.006,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    output7 = [0.007,0.002,0.003,0.004,0.002,0.003,0.004,0.005,0.005,0.005]
    """ results=[output1, output2, output3, output4, output5, output6, output7] """
    return jsonify(results=[output1, output2, output3, output4, output5, output6, output7])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
