# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import time


class Denosing_AutoEncoder():
    def __init__(self, hidden_size, input_data, epochs, learning_rate=0.1, is_training=True, keep_pro=0.7):
        self.hidden_size = hidden_size
        self.input_data = input_data
        self.keep_pro = keep_pro
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.hidden_output = None
        self.w = None
        self.b = None
        self.w_eval = None
        self.b_eval = None

    def fit(self, x, i):
        corrupt = tf.layers.dropout(x, rate=1-self.keep_pro)
        input_size = self.input_data.shape[1]
        w_max = 4 * np.sqrt(6.0 / (input_size + self.hidden_size))

        with tf.variable_scope("layer" + str(i)):
            self.w = tf.get_variable("en-weights", shape=[input_size, self.hidden_size],
                                     initializer=tf.random_uniform_initializer(minval=-w_max, maxval=w_max,
                                                                               dtype=tf.float32))
            self.b = tf.get_variable("en-biases", shape=[self.hidden_size], initializer=tf.constant_initializer(0.0))

            de_w = tf.transpose(self.w, name="de-weights")
            de_b = tf.get_variable("de-biases", shape=[input_size], initializer=tf.constant_initializer(0.0))

        layer_out1 = tf.matmul(corrupt, self.w) + self.b
        layer_act1 = tf.nn.sigmoid(layer_out1)

        layer_out2 = tf.matmul(layer_act1, de_w) + de_b
        layer_act2 = tf.nn.sigmoid(layer_out2)

        return layer_act1, layer_act2

    def train(self, batch_size, i):
        x = tf.placeholder(tf.float32, [None, self.input_data.shape[1]], name="x")

        middle, out = self.fit(x, i)

        loss = tf.reduce_mean(tf.pow(x - out, 2), name="losses")
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)    # 0.01
        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        #tf.summary.scalar("layer-losses", loss)

        nums = self.input_data.shape[0] // batch_size

        #tf.summary.scalar("loss"+str(i), loss)
        #merged = tf.summary.merge_all()
        #summ_writer = tf.summary.FileWriter(summary_logdir)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            print('---------------------------------')

            for i in range(self.epochs):
                for j in range(nums):
                    inputs = self.input_data[j * batch_size: (j + 1) * batch_size, :]
                    _, losses = sess.run([train_step, loss], feed_dict={x: inputs})

                    #if j % 10 == 0:
                        #summ_writer.add_summary(summ, i*nums+j)
                if i % 5 == 0:
                    print(losses)
            #summ_writer.close()

            self.w_eval = self.w.eval()
            self.b_eval = self.b.eval()
            self.hidden_output = middle.eval(feed_dict={x: self.input_data})


    def get_value(self):
        return self.w_eval, self.b_eval, self.hidden_output
        




