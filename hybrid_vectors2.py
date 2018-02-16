#!/usr/bin/python3
import random

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.losses.losses_impl import Reduction

BATCH_SIZE = 2000
HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 50000
DISPLAY_STEP = 100
WINDOW = 2

def main(path):
    ids = read_ids(path + '/external_ids.txt')
    sessions = read_sessions(path + '/sessions.txt', 2)
    features = read_features(path)
    train(sessions, features, ids)

def read_ids(path):
    return [line.strip() for line in open(path, encoding='utf-8')]

def read_sessions(path, min_length):
    sessions = []
    with open(path, 'r') as f:
        for line in f:
            ids = [int(id) for id in line.split()]
            if len(ids) >= min_length:
                sessions.append(ids)
    return sessions

def read_features(path):
    content_ids = [int(line.strip()) for line in open(path + '/content_ids.txt')]
    content_vectors = np.load(path + '/content_vectors.npy')

    nav_ids = [int(line.strip()) for line in open(path + '/nav_ids.txt')]
    nav_vectors = np.load(path + '/nav_vectors.npy')

    max_id = max(content_ids + nav_ids)
    m = content_vectors.shape[1]
    n = nav_vectors.shape[1]
    all_vectors = np.zeros((max_id + 1, m + n))
    all_vectors[content_ids, 0:m] = content_vectors
    all_vectors[nav_ids, m:] = nav_vectors

    return all_vectors

def network(input, reuse):
    layer1 = tf.layers.dense(inputs=input, units=HIDDEN_SIZE * 4, activation=tf.nn.relu, reuse=reuse, name='layer-1')
    layer2 = tf.layers.dense(inputs=layer1, units=HIDDEN_SIZE * 2, activation=tf.nn.relu, reuse=reuse, name='layer-2')
    layer3 = tf.layers.dense(inputs=layer2, units=HIDDEN_SIZE, activation=tf.nn.relu, reuse=reuse, name='layer-3')
    return layer3

def train(sessions, features, ids):
    N, D = features.shape

    # Heavily based on:
    # https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py
    # https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
    # https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings


    # X and Y are the concatenated input embeddings
    X = tf.placeholder(tf.float32, [None, D])
    Y = tf.placeholder(tf.float32, [None, D])
    Z = tf.placeholder(tf.float32, [None, D])

    # E is for examples... it contains all vectors
    E = tf.placeholder(tf.float32, [None, D])

    # We go through a shared embedding and fully connected layer for both X and Y
    X_merged = network(X, False)
    Y_merged = network(Y, True)
    Z_merged = network(Z, True)
    E_merged = network(E, True)

    # Contrastive loss
    margin = 0.2
    d_pos = tf.reduce_sum(tf.square(X_merged - Y_merged), 1)
    d_neg = tf.reduce_sum(tf.square(X_merged - Z_merged), 1)
    loss_fn = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(loss_fn)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    logs_path = '/tmp/tensorflow/rnn_words'
    writer = tf.summary.FileWriter(logs_path)

    # Input x and y are the actual feeding pipelines
    input_X = np.zeros([BATCH_SIZE, D], dtype='float32')
    input_Y = np.zeros([BATCH_SIZE, D], dtype='float32')
    input_Z = np.zeros([BATCH_SIZE, D], dtype='float32')

    with tf.Session() as session:
        session.run(init)
        loss_total = 0

        writer.add_graph(session.graph)

        for epoch in range(EPOCHS):
            fetch_batch(sessions, features, input_X, input_Y, input_Z)

            _, loss = session.run([optimizer, loss_fn], feed_dict={X: input_X, Y: input_Y, Z: input_Z})
            loss_total += loss
            if (epoch+1) % DISPLAY_STEP == 0:
                print("Iter= " + str(epoch+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/DISPLAY_STEP))
                loss_total = 0
                E_vecs, = session.run([E_merged], feed_dict={E : features})
                for word in ('dog', 'cat', 'saxophone'):
                    i = ids.index('simple.wikipedia:' + word)
                    assert(i >= 0)
                    target = E_vecs[i, :]
                    dist_2 = np.sum((target - E_vecs) ** 2, axis=1)
                    closest_indexes = dist_2.argsort()[:5]
                    closest_words = [ids[i] for i in closest_indexes]
                    print('\tknn for %s: %s' % (word, ', '.join(closest_words)))


                    # dots =

                # symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                # symbols_out = training_data[offset + n_input]
                # symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                # print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))

        print("Optimization Finished!")
        # print("Elapsed time: ", elapsed(time.time() - start_time))
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (logs_path))
        print("Point your web browser to: http://localhost:6006/")
        # while True:
        #     prompt = "%s words: " % n_input
        #     sentence = input(prompt)
        #     sentence = sentence.strip()
        #     words = sentence.split(' ')
        #     if len(words) != n_input:
        #         continue
        #     try:
        #         symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
        #         for i in range(32):
        #             keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        #             onehot_pred = session.run(pred, feed_dict={x: keys})
        #             onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        #             sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
        #             symbols_in_keys = symbols_in_keys[1:]
        #             symbols_in_keys.append(onehot_pred_index)
        #         print(sentence)
        #     except:
        #         print("Word not in dictionary")


def fetch_batch(sessions, features, X, Y, Z):
    (N, D) = features.shape
    assert(X.shape == (BATCH_SIZE, D))
    assert(Y.shape == (BATCH_SIZE, D))
    assert(Z.shape == (BATCH_SIZE, D))

    random.shuffle(sessions)

    for i, s in enumerate(sessions[:BATCH_SIZE]):
        # Focus item is in session at index j
        j = random.randint(0, len(s) - 1)

        # Related item is at session index k
        start = max(0, j - WINDOW)
        end = min(len(s), j + WINDOW)
        k = random.choice([x for x in range(start, end) if x != j])

        # Unrelated item is id m
        while True:
            m  = random.randint(0, N-1)
            if m not in s:
                break

        X[i,:] = features[s[j],:]
        Y[i,:] = features[s[k],:]
        Z[i,:] = features[m]




if __name__ == '__main__':
    main('./dataset')