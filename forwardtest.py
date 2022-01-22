import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument("-m1", default=1, type=float)
parser.add_argument("-m2", default=1, type=float)
parser.add_argument("-m3", default=0, type=float)
parser.add_argument("-r1", default=0.1, type=float)
parser.add_argument("-r2", default=0.1, type=float)
parser.add_argument("-r3", default=0.1, type=float)
parser.add_argument("-r4", default=0.1, type=float)
parser.add_argument("-r5", default=0.1, type=float)
parser.add_argument("-r6", default=0.1, type=float)
args = parser.parse_args()

#加载神经网络模型的权值和偏置值
def load_weights(output_folder,num_layers):
    weights = []
    biases = []
    for i in range(0, num_layers):
        weight_i = np.loadtxt(output_folder+"/w_"+str(i)+".txt",delimiter=',')
        w_i = tf.Variable(weight_i,dtype=tf.float32)
        weights.append(w_i)
        bias_i = np.loadtxt(output_folder+"/b_"+str(i)+".txt",delimiter=',')
        b_i = tf.Variable(bias_i,dtype=tf.float32)
        biases.append(b_i)
    return weights , biases

#加载测试集数据
def get_data(data):
    for i in range(1,17):
        x_file = data+"/input"+str(i)+".csv"
        y_file = data+"/output"+str(i)+".csv"
        test_X_tmp = np.genfromtxt(x_file,delimiter=',')
        test_Y_tmp = np.genfromtxt(y_file,delimiter=',')
        test_X_tmp = test_X_tmp[50000:55000,0:16]
        test_Y_tmp = test_Y_tmp[50000:55000,0:201]
        if i == 1:
            test_X = test_X_tmp
            test_Y = test_Y_tmp
        else:
            test_X = np.concatenate((test_X,test_X_tmp),axis=0)
            test_Y = np.concatenate((test_Y,test_Y_tmp),axis=0)
    return test_X, test_Y          

#神经网络模型
def decoder(X, weights, biases, num_layers, keep_prob):
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



num_layers = 7
test_number = 70500
data = 'data' #样本数据的存储路径
output_folder = 'results/12_layer_films' #神经网络参数的存储路径
weights, biases = load_weights(output_folder,num_layers) #加载神经网络模型的权重和偏置
test_X, test_Y = get_data(data) #加载测试集数据


x_size = test_X.shape[1]
test_size = test_X.shape[0]
y_size = test_Y.shape[1]
keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None,y_size])

#x_input = np.array([[args.m1,args.m2,args.m3,args.r1,args.r2,args.r3,args.r4,args.r5,args.r6]])
x_input = test_X[test_number][0:x_size].reshape(1,x_size)


# Forward propagation
y_hat  = decoder(X, weights,biases, num_layers, keep_prob)
#cost = tf.reduce_sum(tf.square(y-y_hat))
#cost = tf.div(tf.cast(tf.reduce_sum(tf.square(y-y_hat)),dtype=tf.float32),test_size)



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    y = sess.run(y_hat,feed_dict={X:x_input,keep_prob:1})
    #loss = sess.run(cost, feed_dict={X: test_X, y: test_Y, keep_prob:1})

  

sess.close()


outputfile = open('./targetspectrum.csv', 'w')
outputfile1 = open('./predictedspectrum.csv', 'w')

legend = []
y_actual = test_Y[test_number].tolist()   
y_predict = y[0].tolist()


csv_writer = csv.writer(outputfile)
csv_writer1 = csv.writer(outputfile1)
csv_writer.writerow(y_actual)
csv_writer1.writerow(y_predict)
outputfile.close()
outputfile1.close()

print(sum(np.square(np.array(y_actual)-np.array(y_predict))))
legend.append(str('Actual spectrum'))
legend.append(str('Predicted spectrum'))
plt.plot(range(400,1001,3),y_actual)
plt.plot(range(400,1001,3),y_predict)
plt.title('Comparing spectrums')
plt.ylabel("Transmittance")
plt.xlabel("Wavelength (nm)")
plt.legend(legend, loc='lower right', fontsize=9, borderaxespad=2)
plt.xlim((400, 1000))
plt.ylim((-0.01, 1.098))
plt.tick_params(direction='in',pad=6,labelsize=9)
plt.show()

