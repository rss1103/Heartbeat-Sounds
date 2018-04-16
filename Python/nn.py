# Beat tracking example
from __future__ import print_function
import sys
import os
import glob
import ntpath
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#data = pd.read_csv('tempo-fe.csv')
data = pd.read_csv('pca_data.csv')
#X = data.values[:, 3:]


# features, labels = np.empty((0,577)), np.empty(0)#data.values[:, 4:]
#features, labels = np.empty((0,384)), np.empty(0)#data.values[:, 4:]
features, labels = np.empty((0,82)), np.empty(0)#data.values[:, 4:]
#labels = #data.values[:, 3]
labels_map = list(set(data.values[:, 2]))
#print(labels_map)
#exit()


for i in range(len(data.values)):
    labels = np.append(labels, labels_map.index(data.values[i, 2]))
    ext_features = np.hstack([data.values[i, 3:]])
    features = np.vstack([features,ext_features])

#x = features
#features = (x - x.min(0)) / x.ptp(0)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    print(np.unique(labels))
    #exit()
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

features, labels = np.array(features), np.array(labels, dtype = np.int)
labels = one_hot_encode(labels)
  
print(labels)  
#print(len(features[0]))
#print(labels)

#exit()
train_test_split = np.random.rand(len(features)) < 0.80#0.70
train_x = features[train_test_split]
train_y = labels[train_test_split]
test_x = features[~train_test_split]
test_y = labels[~train_test_split]

print("train: %d"%len(train_x))
print("test_x: %d"%len(test_x))

print("train_y: %d"%len(train_y))
print("test_y: %d"%len(test_y))
#Training Neural Network with TensorFlow

import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support

print(features)
print(features.shape)
print(features.shape[1])
training_epochs = 300
n_dim = features.shape[1]
n_classes = len(labels_map)

multiplier = 8
n_hidden_units_1 = int(100 * multiplier)#280
n_hidden_units_2 = int(100 * multiplier)#280
n_hidden_units_3 = int(100 * multiplier)#300
n_hidden_units_4 = int(100 * multiplier)#300
n_hidden_units_5 = int(100 * multiplier)#300
n_hidden_units_6 = int(100 * multiplier)#300
n_hidden_units_7 = int(100 * multiplier)#300
n_hidden_units_8 = int(100 * multiplier)#300
n_hidden_units_9 = int(100 * multiplier)#300
n_hidden_units_10 = int(100 * multiplier)#300

#n_hidden_units_three = 100#300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_1], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_1], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_1, n_hidden_units_2], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_2], mean = 0, stddev=sd))
h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_2, n_hidden_units_3], mean = 0, stddev=sd))
b_3 = tf.Variable(tf.random_normal([n_hidden_units_3], mean = 0, stddev=sd))
h_3 = tf.nn.tanh(tf.matmul(h_2, W_3) + b_3)

W_4 = tf.Variable(tf.random_normal([n_hidden_units_3, n_hidden_units_4], mean = 0, stddev=sd))
b_4 = tf.Variable(tf.random_normal([n_hidden_units_4], mean = 0, stddev=sd))
h_4 = tf.nn.tanh(tf.matmul(h_3, W_4) + b_4)

W_5 = tf.Variable(tf.random_normal([n_hidden_units_4, n_hidden_units_5], mean = 0, stddev=sd))
b_5 = tf.Variable(tf.random_normal([n_hidden_units_5], mean = 0, stddev=sd))
h_5 = tf.nn.tanh(tf.matmul(h_4, W_5) + b_5)

W_6 = tf.Variable(tf.random_normal([n_hidden_units_5, n_hidden_units_6], mean = 0, stddev=sd))
b_6 = tf.Variable(tf.random_normal([n_hidden_units_6], mean = 0, stddev=sd))
h_6 = tf.nn.tanh(tf.matmul(h_5, W_6) + b_6)

W_7 = tf.Variable(tf.random_normal([n_hidden_units_6, n_hidden_units_7], mean = 0, stddev=sd))
b_7 = tf.Variable(tf.random_normal([n_hidden_units_7], mean = 0, stddev=sd))
h_7 = tf.nn.tanh(tf.matmul(h_6, W_7) + b_7)

W_8 = tf.Variable(tf.random_normal([n_hidden_units_7, n_hidden_units_8], mean = 0, stddev=sd))
b_8 = tf.Variable(tf.random_normal([n_hidden_units_8], mean = 0, stddev=sd))
h_8 = tf.nn.tanh(tf.matmul(h_7, W_8) + b_8)

W_9 = tf.Variable(tf.random_normal([n_hidden_units_8, n_hidden_units_9], mean = 0, stddev=sd))
b_9 = tf.Variable(tf.random_normal([n_hidden_units_9], mean = 0, stddev=sd))
h_9 = tf.nn.tanh(tf.matmul(h_8, W_9) + b_9)

W_10 = tf.Variable(tf.random_normal([n_hidden_units_9, n_hidden_units_10], mean = 0, stddev=sd))
b_10 = tf.Variable(tf.random_normal([n_hidden_units_10], mean = 0, stddev=sd))
h_10 = tf.nn.tanh(tf.matmul(h_9, W_10) + b_10)
#W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
#b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
#h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

#W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
#b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
#h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)


W = tf.Variable(tf.random_normal([n_hidden_units_4, n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

global_step = tf.Variable(0)
init = tf.global_variables_initializer()

#cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
# cost_function = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(y_) - (1-Y) * tf.log(1-y_), reduction_indices=[1]))
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# reg_constant = (learning_rate/(2*n_dim))  # Choose an appropriate one.
# loss = cost_function + reg_constant * sum(reg_losses)

# Normal loss function
beta = 0.001
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=train_y))
# Loss function with L2 Regularization with decaying learning rate beta=0.5
regularizers = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2)
               # tf.nn.l2_loss(W_3) + tf.nn.l2_loss(W_4) + \
               # tf.nn.l2_loss(W_5) + tf.nn.l2_loss(W_6) + \
               # tf.nn.l2_loss(W_7) + tf.nn.l2_loss(W_8)

# regularizers *= beta
#loss = tf.reduce_mean(loss + beta * regularizers)
cost_function = loss
# regularizationTerm = (learning_rate/(2*n_dim)) * (sum(tf.square(W_1[1:])) + sum(tf.square(W_2[1:])))
    # J = J + regularizationTerm;
# cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_) - (1-Y) * tf.log(1-y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#vv

# start_learning_rate = 0.5
# learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
cm = {}
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        sys.stdout.write("Epoch #%d   \r" % (epoch))
        sys.stdout.flush()
        # print("here") 
        # print(len(train_x))
        # print(len(train_y))        
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:train_x,Y:train_y})
        cost_history = np.append(cost_history,cost)
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))
    

for i in range(len(y_pred)):
    yy = y_pred[i]
    yyy = y_true[i] 
    #print("yy%d yyy:%d" % (yy, yyy))
    if yy in cm:
        #print(cm)
        cm[yy][yyy] += 1
    else:
        cm[yy] = {0:0, 1:0, 2:0, 3:0, 4:0}
        cm[yy][yyy] = 1

print(cm)
fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print ("F-Score: %f" %round(f,3))

# data = parse_audio_files("data", ["testa"])
# np.save("npdata.npy", data[0])
# print()

# plt.figure()
# plt.subplot(3, 1, 1)
# librosa.display.waveplot(y, sr=sr)
# plt.title('Monophonic')
# plt.show()
# print(sr)
# print(y)


# 3. Run the default beat tracker
# tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
# beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# print('Saving output to beat_times.csv')
# librosa.output.times_csv('beat_times_wav.csv', beat_times)

 


# os.system("pause")
# print("done")
