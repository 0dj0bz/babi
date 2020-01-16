from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def make_header():
    header = "rec_id,"

    data_header = ""

    for i in range(0,20):
        for j in range(0,300):
            data_header += "S"+str(i)+"_"+str(j)+","

    for i in range(0,10):
        for j in range(0,300):
            data_header += "Q"+str(i)+"_"+str(j)+","

    for i in range(0,1):
        for j in range(0,300):
            data_header += "A"+str(i)+"_"+str(j)
            if j < 299:
            	data_header += ","

    header += data_header

    return header



train_file_path = os.path.join('train1_2.csv')

df = pd.read_csv(train_file_path)

df_target = df.loc[:,'A0_0':'A0_299']
df_values = df.loc[:,'S0_0':'Q9_299']


dataset = tf.data.Dataset.from_tensor_slices((df_values.values, df_target.values)).batch(15).repeat()


inputs = tf.keras.Input(shape=(9000,))
# x = layers.Dense(10, input_shape=(3000,), activation='relu')(inputs)
x = layers.Reshape((30,300))(inputs)
lstm_layer1 = layers.LSTM(300)(x)
outputs = tf.keras.layers.Dense(300)(lstm_layer1)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model = models.Sequential()
# model.add(layers.Dense(10, input_shape=(3000,), activation='relu'))
# model.add(layers.Dense(20, activation='relu'))
# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='cosine_similarity',
              metrics=['accuracy'])

history = model.fit(x=dataset, epochs=300, steps_per_epoch=100, verbose=2, shuffle=False)

test_file_path = os.path.join('test1_2.csv')

df2 = pd.read_csv(test_file_path)

df2_target = df2.loc[:,'A0_0':'A0_299']
df2_values = df2.loc[:,'S0_0':'Q9_299']

dataset2 = tf.data.Dataset.from_tensor_slices((df2_values.values, df2_target.values)).batch(1000).repeat()

model.evaluate(x=dataset2, steps=1, verbose=2)




