#!/usr/bin/python3
import random

import tensorflow as tf
import numpy as np

BATCH_SIZE = 1000
TIME_STEPS = 4
HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 50
DISPLAY_STEP = 1

def main(path):
    sessions = read_sessions(path + '/sessions.txt', TIME_STEPS + 1)
    features = read_features(path)
    train(sessions, features)

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
    print(content_vectors.shape, nav_vectors.shape, all_vectors.shape)
    all_vectors[content_ids, 0:m] = content_vectors
    all_vectors[nav_ids, m:] = nav_vectors

    return all_vectors

def train(sessions, features):
    N, D = features.shape

    # Heavily based on:
    # https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py
    # https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537
    # https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings


    # X is the input
    X = tf.placeholder(tf.float32, [TIME_STEPS, BATCH_SIZE, D])
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, N])

    input_X = np.zeros([TIME_STEPS, BATCH_SIZE, D], dtype='float32')
    input_Y = np.zeros([BATCH_SIZE, N], dtype='float32')

    weights = tf.Variable(tf.random_normal([HIDDEN_SIZE, N]))
    biases = tf.Variable(tf.random_normal([N]))


    lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
    X_list = tf.unstack(X, num=TIME_STEPS, axis=0)
    outputs, states = tf.contrib.rnn.static_rnn(lstm, X_list, dtype=tf.float32)
    pred = tf.matmul(outputs[-1], weights) + biases

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    logs_path = '/tmp/tensorflow/rnn_words'
    writer = tf.summary.FileWriter(logs_path)

    with tf.Session() as session:
        session.run(init)
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)

        for epoch in range(EPOCHS):
            fetch_batch(sessions, features, input_X, input_Y)

            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred],
                                                    feed_dict={X: input_X, Y: input_Y})
            print(np.argmax(onehot_pred, axis=1))
            print(np.argmax(input_Y, axis=1))
            loss_total += loss
            acc_total += acc
            if (epoch+1) % DISPLAY_STEP == 0:
                print("Iter= " + str(epoch+1) + ", Average Loss= " + \
                      "{:.6f}".format(loss_total/DISPLAY_STEP) + ", Average Accuracy= " + \
                      "{:.2f}%".format(100*acc_total/DISPLAY_STEP))
                acc_total = 0
                loss_total = 0
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

def fetch_batch(sessions, features, X, Y):
    N, D = features.shape
    assert(X.shape == (TIME_STEPS, BATCH_SIZE, D))
    assert(Y.shape == (BATCH_SIZE, N))
    Y[:,:] = 0

    random.shuffle(sessions)

    for i, s in enumerate(sessions[:BATCH_SIZE]):
        start = random.randint(0, len(s) - (TIME_STEPS + 1))
        end = start + TIME_STEPS
        for j, index in enumerate(s[start:end]):
            X[j,i,:] = features[index,:]
        Y[i][s[end]] = 1

    return (X, Y)




if __name__ == '__main__':
    main('./dataset')