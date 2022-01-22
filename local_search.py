# coding: utf-8
import tensorflow as tf
import time
import os
import random
import numpy as np
import math


#神经网络模型
def decoder_local(X, weights, biases, num_layers, keep_prob):
    htemp = None
    for i in range(0, num_layers):
        if i==0:
            htemp = tf.sin(tf.add(tf.matmul(X,weights[i]),biases[i]))
            print(htemp)
        elif i==1:   
            htemp =  tf.nn.dropout(tf.sin(tf.add(tf.matmul(htemp,weights[i]),biases[i])), keep_prob)    
        elif i==2:  
            htemp = tf.nn.dropout(tf.sin(tf.add(tf.matmul(htemp,weights[i]),biases[i])), keep_prob)
        elif i==3:  
            htemp = tf.nn.dropout(tf.sin(tf.add(tf.matmul(htemp,weights[i]),biases[i])), keep_prob)
        elif i==4:  
            htemp = tf.nn.dropout(tf.sin(tf.add(tf.matmul(htemp,weights[i]),biases[i])), keep_prob)
        elif i==5:  
            htemp = tf.nn.dropout(tf.sin(tf.add(tf.matmul(htemp,weights[i]),biases[i])), keep_prob)
        else:   
            yval = tf.add(tf.matmul(htemp,weights[i]),biases[i])
    return yval


#梯度优化算法的执行过程
def match_spectrum(X_material, X_thickness, num_layers, weights, biases, y_size, batch_y):


    init_list_rand_material = tf.constant(X_material,dtype=tf.float32)
    init_list_rand_thickness = tf.constant(X_thickness,dtype=tf.float32)
    x_thickness = tf.Variable(init_list_rand_thickness)
    keep_prob = tf.placeholder(tf.float32)
    y = tf.placeholder("float", shape=[None,y_size])
    x_total = tf.concat([init_list_rand_material, x_thickness], 1)

    
    y_bp = decoder_local(x_total, weights,biases,num_layers, keep_prob)
    extra_cost_bp = tf.reduce_sum(tf.square(tf.minimum(x_thickness*100,0)) + tf.square(tf.maximum(x_thickness*100,100)-100)) #约束函数
    cost_bp = tf.reduce_sum(tf.square(y-y_bp))+extra_cost_bp #均方差函数+约束函数
    #cost_bp = tf.reduce_sum(tf.square(y[:,0:120]-y_bp[:,0:120]))+tf.multiply(tf.reduce_sum(tf.square(y[:,120:180]-y_bp[:,120:180])),30)+tf.reduce_sum(tf.square(y[:,180:y_size]-y_bp[:,180:y_size]))+5.0*extra_cost_bp*extra_cost_bp
    global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(0.001,global_step,100,0.99,staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.8,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam').minimize(cost_bp,global_step=global_step,var_list=[x_thickness])
    bp_loss =0; prev_losses =0; limit_num = 0; step_bp = 0;

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        while step_bp < 1000: #梯度优化算法每次最大的更新次数1000
            sess.run(optimizer, feed_dict={y: batch_y, keep_prob:1})
            bp_loss = sess.run(cost_bp, feed_dict={y: batch_y, keep_prob:1})#batch_y
            step_bp += 1
            if (step_bp % 10 == 0 or step_bp == 1):
                print("Step_bp: " + str(step_bp) + " : Loss: " + str(bp_loss) +" : predicted structure: " + str(x_total.eval()))
            if (abs(bp_loss-prev_losses) < .001 and bp_loss >= 1):
                limit_num += 1
                if limit_num > 500:
                   limit_num=0
                   prev_losses = bp_loss
                   break
            elif bp_loss < 0.1:
                limit_num=0
                prev_losses = bp_loss
                break
            else:
                limit_num = 0
            prev_losses = bp_loss
            bp_loss = 0
        predicted_structure = x_total.eval()
    sess.close()
    return prev_losses, predicted_structure

