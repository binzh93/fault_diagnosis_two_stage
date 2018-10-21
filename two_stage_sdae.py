# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import time
from DAE import *
import os

# # os.chdir("/Users/bin/Desktop/fault diagnosis/毕业论文相关/未命名文件夹/fault diagnosis/fault_data")
# os.chdir("/workspace/mnt/group/face-reg/zhubin/fault_diagnosis/fault_data")

def Fully_connected(x, units=10, layer_name='fully_connected') :
  with tf.name_scope(layer_name):
      return tf.layers.dense(inputs=x, use_bias=True, units=units)

class Stacked_Denoising_AutoEncoder_Two_Stage():
    def __init__(self, 
                train_X_up, 
                train_X_down,
                train_Y, 
                test_X_up,
                test_X_down,
                test_Y,
                inside_epochs, 
                inside_batch_size, 
                outside_epochs, 
                outside_train_batch_size,
                outside_test_batch_size,
                inside_learning_rate, 
                learning_rate, 
                layer_list_up, 
                layer_list_down,
                nclass, 
                moving_decay):
        self.train_X_up = train_X_up
        self.train_X_down = train_X_down
        self.train_Y = train_Y
        self.test_X_up = test_X_up
        self.test_X_down = test_X_down
        self.test_Y = test_Y
        self.nclass = nclass
        self.inside_epochs = inside_epochs
        self.inside_batch_size = inside_batch_size
        self.outside_epochs = outside_epochs
        self.outside_train_batch_size = outside_train_batch_size
        self.outside_test_batch_size = outside_test_batch_size
        self.learning_rate = learning_rate
        self.inside_learning_rate = inside_learning_rate
        self.layer_list_up = layer_list_up
        self.layer_list_down = layer_list_down
        self.w_val_up = []
        self.b_val_up = []
        self.w_val_down = []
        self.b_val_down = []

    def fit_up(self, x, inside_batch_size):
        next_input_data = self.train_X_up
        for i in range(len(self.layer_list_up)):
            dae = Denosing_AutoEncoder(self.layer_list_up[i], 
                                        next_input_data, 
                                        epochs=self.inside_epochs,     # 30
                                        learning_rate=self.inside_learning_rate)   
            dae.train(inside_batch_size, i)
            w, b, middle_out = dae.get_value()
            self.w_val_up.append(w)
            self.b_val_up.append(b)
            next_input_data = middle_out

        layer_out = x
        for i in range(len(self.layer_list_up)-1):
            if i == 0:
                weights = tf.get_variable("weights" + str(i), shape=[self.train_X_up.shape[1], self.layer_list_up[i]],
                                          initializer=tf.constant_initializer(self.w_val_up[i]))
            else:
                weights = tf.get_variable("weights" + str(i), shape=[self.layer_list_up[i - 1], self.layer_list_up[i]],
                                          initializer=tf.constant_initializer(self.w_val_up[i]))
            biases = tf.get_variable("biases" + str(i), shape=[self.layer_list_up[i]],
                                initializer=tf.constant_initializer(self.b_val_up[i]))
            layer_out = tf.nn.relu(tf.matmul(layer_out, weights) + biases)

        weights = tf.get_variable("weights-end", shape=[self.layer_list_up[len(self.layer_list_up)-2], self.layer_list_up[-1]],
                                  initializer=tf.constant_initializer(self.w_val_up[-1]))
        biases = tf.get_variable("biases-end", shape=[self.layer_list_up[-1]], initializer=tf.constant_initializer(self.b_val_up[-1]))

        out = tf.matmul(layer_out, weights) + biases
        return out


    def fit_down(self, x, inside_batch_size):
        next_input_data = self.train_X_down
        for i in range(len(self.layer_list_down)):
            dae = Denosing_AutoEncoder(self.layer_list_down[i], 
                                        next_input_data, 
                                        epochs=self.inside_epochs,     # 30
                                        learning_rate=self.inside_learning_rate)   
            dae.train(inside_batch_size, i)
            w, b, middle_out = dae.get_value()
            self.w_val_down.append(w)
            self.b_val_down.append(b)
            next_input_data = middle_out

        layer_out = x
        for i in range(len(self.layer_list_down)-1):
            if i == 0:
                weights = tf.get_variable("weights" + str(i), shape=[self.train_X_down.shape[1], self.layer_list_down[i]],
                                          initializer=tf.constant_initializer(self.w_val_down[i]))
            else:
                weights = tf.get_variable("weights" + str(i), shape=[self.layer_list_down[i - 1], self.layer_list_down[i]],
                                          initializer=tf.constant_initializer(self.w_val_down[i]))
            biases = tf.get_variable("biases" + str(i), shape=[self.layer_list_down[i]],
                                initializer=tf.constant_initializer(self.b_val_down[i]))
            layer_out = tf.nn.relu(tf.matmul(layer_out, weights) + biases)

        weights = tf.get_variable("weights-end", shape=[self.layer_list_down[len(self.layer_list_down)-2], self.layer_list_down[-1]],
                                  initializer=tf.constant_initializer(self.w_val_down[-1]))
        biases = tf.get_variable("biases-end", shape=[self.layer_list_down[-1]], initializer=tf.constant_initializer(self.b_val_down[-1]))

        out = tf.matmul(layer_out, weights) + biases
        return out

    def train(self):
        x1 = tf.placeholder(tf.float32, shape=[None, self.train_X_up.shape[1]], name="x1")
        x2 = tf.placeholder(tf.float32, shape=[None, self.train_X_down.shape[1]], name="x2")
        y = tf.placeholder(tf.float32, shape=[None, self.train_Y.shape[1]], name="label")
        with  tf.variable_scope("stage1_up"):
            out1 = self.fit_up(x1, inside_batch_size=128)
        with  tf.variable_scope("stage1_down"):
            out2 = self.fit_down(x2, inside_batch_size=128)
        alpha1 = 0.95
        alpha2 = 0.05
        out_merge = alpha1 * out1 + alpha2 * out2
        fc1_ = Fully_connected(out_merge, units=300, layer_name="fc1_second_stage")
        relu1_ = tf.nn.relu(fc1_)
        fc2_ = Fully_connected(relu1_, units=self.nclass, layer_name="fc2_second_stage")

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc2_)
        losses = tf.reduce_mean(cross_entropy)

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(losses)
        prediction = tf.argmax(fc2_, 1)

        correct_nums = tf.equal(prediction, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_nums, dtype=tf.float32), name="accuracy")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            total_epochs = self.outside_epochs
            train_batch_size = self.outside_train_batch_size

            assert self.train_X_up.shape[0] == self.train_X_down.shape[0]
            total_nums = self.train_X_up.shape[0]
            train_iteration = total_nums / train_batch_size

            for it1 in range(total_epochs):
                # train the network
                train_loss = 0.0
                train_acc = 0.0
                for it2 in range(train_iteration):
                    batch_x_up = self.train_X_up[it2*train_batch_size: (it2+1)*train_batch_size, :]
                    batch_x_down = self.train_X_down[it2*train_batch_size: (it2+1)*train_batch_size, :]
                    batch_y = self.train_Y[it2*train_batch_size: (it2+1)*train_batch_size, :]
                    _, batch_loss, batch_acc = sess.run([train_step, losses, accuracy], 
                                                        feed_dict={x1: batch_x_up, x2: batch_x_down, y: batch_y})
                    train_loss += batch_loss
                    train_acc += batch_acc

                ave_loss = train_loss*1.0/train_iteration
                ave_acc = train_acc*1.0/train_iteration

                # test in every epoch
                test_batch_size = self.outside_test_batch_size
                assert self.test_X_up.shape[0] == self.test_X_down.shape[0]
                test_iteration = self.test_X_up.shape[0] / test_batch_size
                test_acc = 0.0
                test_loss = 0.0
                for it3 in range(test_iteration):
                    test_batch_x_up = self.test_X_up[it3*test_batch_size: (it3+1)*test_batch_size, :]
                    test_batch_x_down = self.test_X_down[it3*test_batch_size: (it3+1)*test_batch_size, :]
                    test_batch_y = self.test_Y[it3*test_batch_size: (it3+1)*test_batch_size, :]

                    test_batch_loss, test_batch_acc = sess.run([losses, accuracy], 
                                                                feed_dict={x1: test_batch_x_up, x2: test_batch_x_down, y:test_batch_y})
                    test_loss += test_batch_loss
                    test_acc += test_batch_acc
                test_ave_loss = test_loss*1.0/test_iteration
                test_ave_acc = test_acc*1.0/test_iteration

                print("epoch {}, train loss: {}, test acc: {}".format(it1, ave_loss, test_ave_acc))
  






