import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

all_tweets = pd.read_csv('data/emojis_homemade.csv')
tweets = all_tweets.groupby('emoji').filter(lambda c:len(c) > 1000)
train_tweets = tweets[0:100000]
x_train = train_tweets["text"]

text = "I am happy"
tokenizer = Tokenizer(5000)
tokenizer.fit_on_texts (x_train)
#print(tokenizer.word_index)
#x_train_one_hot = tokenizer.texts_to_sequences(x_train)
x_train_one_hot = tokenizer.texts_to_matrix(x_train, mode='binary')
print(x_train_one_hot.shape)
print(x_train_one_hot)

""" model = load_model('./model/linear_classifier.h5')
preds = model.predict(x_train_one_hot)
"""