'''
def SQNL(inputs):
    case_1 = tf.cast(tf.math.greater(inputs, 2), 'float32')
    case_2 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * tf.cast(tf.math.less_equal(inputs, 2), 'float32') * (inputs - tf.math.pow(inputs, 2) / tf.cast(4, 'float32'))
    case_3 = tf.cast(tf.math.greater_equal(inputs, -2), 'float32') * tf.cast(tf.math.less(inputs, 0), 'float32') * (inputs + tf.math.pow(inputs, 2) / tf.cast(4, 'float32'))
    case_4 = tf.cast(tf.math.less(inputs, -2), 'float32') * tf.cast(-1, 'float32')
    return case_1 + case_2 + case_3 + case_4


def SoftExponential(inputs,alpha=1):
    condition_1 = tf.cast(tf.math.less(alpha, 0), 'float32')
    condition_2 = tf.cast(tf.math.equal(alpha, 0), 'float32')
    condition_3 = tf.cast(tf.math.greater(alpha, 0), 'float32')
    case_1 = condition_1 * (-1 / alpha) * tf.math.log(1 - alpha * (inputs + alpha))
    case_2 = condition_2 * inputs
    case_3 = condition_3 * (alpha + (1 / alpha) * (tf.math.exp(alpha * inputs) - 1))
    return case_1 + case_2 + case_3
'''

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import argparse


tf.set_random_seed(42) #伪随机数

#权重初始化函数
def init_weights(shape,stddev=.01):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(weights)

#偏置初始化函数
def init_bias(shape, stddev=0.1):
    """ Weight initialization """
    biases = tf.random_normal([shape], stddev=stddev)
    return tf.Variable(biases)

#权重和偏置值保存函数
def save_weights(weights,biases,output_folder,num_layers):
    for i in range(0, num_layers):
        weight_i = weights[i].eval()
        np.savetxt(output_folder+"/w_"+str(i)+".txt",weight_i,delimiter=',')
        bias_i = biases[i].eval()
        np.savetxt(output_folder+"/b_"+str(i)+".txt",bias_i,delimiter=',')
    return


#prelu激活函数
def prelu(_x, name):
    """parametric ReLU activation"""
    _alpha = tf.get_variable(name + "prelu", shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.5))
    pos = tf.nn.relu(_x)
    neg = _alpha * (_x - tf.abs(_x)) * 0.5

    return pos + neg

#BentIdentity激活函数
def BentIdentity(inputs):
    return inputs + (tf.sqrt(tf.pow(inputs, 2) + 1) - 1) / 2

#CELU激活函数
def CELU(inputs,alpha=6):
    case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs
    case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * alpha * (tf.math.exp(inputs / alpha) - 1)
    return case_1 + case_2

#SCELU激活函数
def SCELU(inputs,alpha=25):
    case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * alpha * (1-tf.math.exp(-(inputs / alpha)))
    case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * alpha * (tf.math.exp(inputs/ alpha) - 1)
    return case_1 + case_2

#ELU激活函数
def ELU(inputs,alpha=5):
    case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * alpha * (1-tf.math.exp(-(inputs)))
    case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * alpha * (tf.math.exp(inputs) - 1)
    return case_1 + case_2

#ETanh激活函数
def ETanh(inputs,beta=8):
    return beta * inputs * tf.math.tanh(inputs)

#SReLU激活函数
def SReLU(inputs, t=-1, a=0.25, r=2, l=1):
    t = tf.cast(t, 'float32')
    a = tf.cast(a, 'float32')
    r = tf.cast(r, 'float32')
    l = tf.cast(l, 'float32')
    condition_1 = tf.cast(tf.math.greater_equal(inputs, tf.math.pow(t, r)), 'float32')
    condition_2 = tf.cast(tf.math.greater(tf.math.pow(t, r), inputs), 'float32') + tf.cast(tf.math.greater(inputs, tf.math.pow(t,l)),'float32')
    condition_3 = tf.cast(tf.math.less_equal(inputs, tf.math.pow(t, l)), 'float32')
    case_1 = condition_1 * (tf.math.pow(t, r) + tf.math.pow(a, r) * (inputs - tf.math.pow(t, r)))
    case_2 = condition_2 * inputs
    case_3 = condition_3 * (tf.math.pow(t, l) + tf.math.pow(a, l) * (inputs - tf.math.pow(t, l)))
    return case_1 + case_2 + case_3



#加载训练集和验证集数据函数
def get_data(data,percentTest=.1,random_state=42):
    for i in range(1,17):
        x_file = data+"/input"+str(i)+".csv"
        y_file = data+"/output"+str(i)+".csv"
        train_X_tmp = np.genfromtxt(x_file,delimiter=',')
        train_Y_tmp = np.genfromtxt(y_file,delimiter=',')
        train_X_tmp = train_X_tmp[0:1000,0:16]
        train_Y_tmp = train_Y_tmp[0:1000,0:201]
        if i == 1:
            train_X_local = train_X_tmp
            train_Y_local = train_Y_tmp
        else:
            train_X_local = np.concatenate((train_X_local,train_X_tmp),axis=0)
            train_Y_local = np.concatenate((train_Y_local,train_Y_tmp),axis=0)
    X_train, X_val, y_train, y_val = train_test_split(train_X_local,train_Y_local,test_size=float(percentTest),random_state=random_state)
    return X_train, y_train, X_val, y_val          


#神经网络模型
def decoder(X, weights, biases, num_layers, keep_prob):
    htemp = None
    for i in range(0, num_layers):
        if i==0:
            htemp = tf.sin(tf.add(tf.matmul(X,weights[i]),biases[i]))
        elif i==1:   
            htemp = tf.nn.dropout(tf.sin(tf.add(tf.matmul(htemp,weights[i]),biases[i])), keep_prob)    
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



numEpochs = 1000 #epochs总的大小
num_layers = 7 #神经网络的隐藏层数+输出层
n_batch = 100 #批处理大小
lr_rate = .0005 #初始学习率
data = 'data' #样本数据的存储路径
output_folder = 'results/12_layer_films' #神经网络参数保存的路径
train_log_dir = './logs/train' #训练过程训练误差保存的路径
test_log_dir = './logs/test' #训练过程验证误差保存的路径
train_X, train_Y, val_X, val_Y = get_data(data) #加载训练数据和验证数据


x_size = train_X.shape[1]
val_size = val_X.shape[0]
y_size = train_Y.shape[1]

val_number = val_size/n_batch
print(x_size)
print(val_size)
print(y_size)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


weights = []
biases = []
# 权重和偏置初始化
for i in range(0,num_layers):
    if i==0:
        weights.append(init_weights((x_size, 250)))
        biases.append(init_bias(250))
    elif i==1:
        weights.append(init_weights((250,500)))
        biases.append(init_bias(500))
    elif i==2:
        weights.append(init_weights((500,500)))
        biases.append(init_bias(500))
    elif i==3:
        weights.append(init_weights((500,500)))
        biases.append(init_bias(500))
    elif i==4:
        weights.append(init_weights((500,500)))
        biases.append(init_bias(500))
    elif i==5:
        weights.append(init_weights((500,250)))
        biases.append(init_bias(250))
    else:
        weights.append(init_weights((250,y_size)))
        biases.append(init_bias(y_size))
    #else:
    #    weight_i = np.loadtxt(output_folder+"/w_"+str(i)+".txt",delimiter=',')
    #    w_i = tf.Variable(weight_i,dtype=tf.float32)
    #    weights.append(w_i)
    #    bias_i = np.loadtxt(output_folder+"/b_"+str(i)+".txt",delimiter=',')
    #    b_i = tf.Variable(bias_i,dtype=tf.float32)
    #    biases.append(b_i)


keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None,y_size])


v_weight = [weights[0], weights[1], weights[2], weights[3], weights[4], weights[5], weights[6], biases[0], biases[1], biases[2], biases[3], biases[4], biases[5], biases[6]]


# Forward propagation
y_hat = decoder(X, weights,biases,num_layers, keep_prob)



# Backward propagation
with tf.name_scope('loss'):
    cost =  tf.div(tf.cast(tf.reduce_sum(tf.square(y-y_hat)),dtype=tf.float32),n_batch)
tf.summary.scalar('per_batch_lost', cost)

global_step = tf.Variable(0, trainable=False)
print("LR Rate: " , lr_rate)
print(int(train_X.shape[0]/n_batch))

learning_rate = tf.train.exponential_decay(lr_rate,global_step,int(train_X.shape[0]/n_batch),.99,staircase=False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.8,beta2=0.999,epsilon=1e-08,use_locking=False,name='Adam').minimize(cost,global_step=global_step,var_list=v_weight)

#Now do the training. 
step =0; curEpoch =0; cum_loss =0; perinc = 0;

merge = tf.summary.merge_all()
start_time=time.time()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer_train = tf.summary.FileWriter(logdir=train_log_dir, graph=sess.graph)
    writer_test = tf.summary.FileWriter(logdir=test_log_dir)

    print("========                         Iterations started                  ========")
    while curEpoch < numEpochs:
        batch_x = train_X[step * n_batch : (step+1) * n_batch]
        batch_y = train_Y[step * n_batch : (step+1) * n_batch]
        summary_train,cuminc, _ = sess.run([merge,cost,optimizer], feed_dict={X: batch_x, y: batch_y, keep_prob:1})
        cum_loss += cuminc
        step += 1
        #End of each epoch. 
        if step ==  int(train_X.shape[0]/n_batch): 
            curEpoch +=1            
            cum_loss = cum_loss/float(step)
            step = 0
            # Every 10 epochs, do a validation. 
            if (curEpoch % 10 == 0 or curEpoch == 1):   
                step_val = np.random.randint(1,val_number)
                batch_val_x = val_X[step_val * n_batch : (step_val+1) * n_batch]
                batch_val_y = val_Y[step_val * n_batch : (step_val+1) * n_batch]
                val_loss,summary_test = sess.run([cost,merge],feed_dict={X:batch_val_x,y:batch_val_y, keep_prob:1})
                writer_train.add_summary(summary_train, global_step=curEpoch)
                writer_test.add_summary(summary_test, global_step=curEpoch)
                print("Validation loss: " , str(val_loss) )
                print("Epoch: " + str(curEpoch+1) + " : Loss: " + str(cum_loss) )
            cum_loss = 0
            perinc = 0
    save_weights(weights,biases,output_folder,num_layers) #训练结束后保存神经网络模型参数
    writer_train.close()
    writer_test.close()
    print(sess.run(learning_rate))
print("========Iterations completed in : " + str(time.time()-start_time) + " ========")
sess.close()


