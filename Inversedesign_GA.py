# coding: utf-8
import tensorflow as tf
import time
import os
import random
import numpy as np
import math
import heapq
from local_search import *

''''
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.Session(config=config)
'''


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
    for i in range(1,17): #16种由不同材料组成的数据
        x_file = data+"/input"+str(i)+".csv"
        y_file = data+"/output"+str(i)+".csv"
        test_X_tmp = np.genfromtxt(x_file,delimiter=',')
        test_Y_tmp = np.genfromtxt(y_file,delimiter=',')
        test_X_tmp = test_X_tmp[50000:55000,0:7]
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


#随机生成初始种群
def generate_population():
    population = []
    for i in range(num_total):
        x=list(np.random.randint(low=0,high=2,size=binary_len_total))
        p = x[:]
        while (p in population):
            x=list(np.random.randint(low=0,high=2,size=binary_len_total))
            p = x[:]
        population.append(p)
    return population

#解码，厚度参数由二进制转换成实数
def decode(popular_gene):
    x_real_value = []
    for i in range(num_total):
        x_real_value_tmp = []
        for j in range(numOfinput):
            if j == 0:
                x_real_value_tmp.extend(popular_gene[i][0:binary_len_material])
            else:
                x_binary_thickness = ''
                x_binary_thickness_tmp = popular_gene[i][binary_len_material+(j-1)*binary_len_thickness:binary_len_material+j*binary_len_thickness]
                for k in range(binary_len_thickness):
                    x_binary_thickness += str(x_binary_thickness_tmp[k])
                x_real_value_tmp.append(int(x_binary_thickness,2)/(2**5-1)) 
        x_real_value.append(x_real_value_tmp)
    return x_real_value


# 遗传算法选择操作
def selection(popular_gene, fitnessvalues):
    population_fitness = []
    individual_fitness = 0
    for i in range(num_total):
        individual_fitness=fitnessvalues[i]+individual_fitness
        population_fitness.append(individual_fitness)
    selection_population=[]
    for p in range(num_total):
        rand=random.uniform(0,population_fitness[num_total-1])
        for j in range(num_total):
            if (rand <= population_fitness[j]):
                selection_population.append(popular_gene[j])
                break
            else:
                continue
    return selection_population

# 遗传算法交叉操作
def crossover(selection_population, Pc=0.9): #Pc=0.8
    crossover_population = []
    random_pair_index = list(np.arange(num_total))
    random.shuffle(random_pair_index)
    for c in range(0,num_total,2):
        slice_index = sorted(random.sample(range(binary_len_total),CR_site))
        crossover_individual_one = selection_population[random_pair_index[c]]
        crossover_individual_two = selection_population[random_pair_index[c+1]]
        if random.uniform(0, 1) <= Pc:
            for i in range(0,CR_site,2):
                crossover_individual_one[slice_index[i]:slice_index[i+1]],crossover_individual_two[slice_index[i]:slice_index[i+1]] = crossover_individual_two[slice_index[i]:slice_index[i+1]],crossover_individual_one[slice_index[i]:slice_index[i+1]]
        crossover_population.append(crossover_individual_one)
        crossover_population.append(crossover_individual_two)
    return crossover_population


# 遗传算法变异操作
def mutation(crossover_population, Pm=0.01): #Pm=0.01
    for m in range(num_total):
        for i in range(binary_len_total):
            is_variation = random.uniform(0, 1)
            if is_variation <= Pm:
                if crossover_population[m][i] == 0:
                    crossover_population[m][i] = 1
                else:
                    crossover_population[m][i] = 0
    return crossover_population






numOfinput = 4 #层数+1
binary_len_material = 4 #两种材料参数的二进制编码长度
binary_len_thickness = 5 #每一层厚度的二进制编码长度
binary_len_total = binary_len_material + (numOfinput-1)*binary_len_thickness #种群中个体总的二进制编码长度
num_total = 100 #随机生成的初始解的总数
CR_site = 6 #指定了6个交换点用于父本的基因交换重组





data = 'data' #样本数据的存储路径
num_layers = 7
output_folder = 'results/3_layer_particles' #神经网络参数的存储路径
weights, biases = load_weights(output_folder,num_layers) #加载神经网络模型的权重和偏置
test_X, test_Y = get_data(data) #加载测试集数据

x_size = test_X.shape[1]
y_size = test_Y.shape[1]

number = 18873; 
x_input = test_X[number].reshape(1,x_size) 
y_input1 = test_Y[number].reshape(1,y_size) 
outputfile = open(output_folder + '/outputofGA.txt', 'w') #新建一个TXT文档用于保存优化算法的输出数据
#testY = np.genfromtxt('onemodegauss.csv',delimiter=',')
testY = np.genfromtxt('Lorenrzian.csv',delimiter=',') #加载任意形状的目标光谱
testY = testY[0:201]
y_input2 = testY.reshape(1,201)
batch_y = y_input1

keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[None,y_size])

y_hat = decoder(X, weights,biases,num_layers, keep_prob) #神经网络模型的输出值

cost = tf.reduce_sum(tf.square(y-y_hat)) #损失函数
#cost = tf.reduce_sum(tf.square(y[:,0:120]-y_hat[:,0:120]))+tf.multiply(tf.reduce_sum(tf.square(y[:,120:180]-y_hat[:,120:180])),30)+tf.reduce_sum(tf.square(y[:,180:y_size]-y_hat[:,180:y_size]))

np_list = generate_population() #初始种群
adp_max_everygeneration = []
adp_max_mean = []
optimum_codition = 0
current_localsearch_loss = 1000
start_time=time.time()
#GA表示genetic algorithm, BP表示back-propagation
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    print("========                         Inversedesign started                  ========")
    for step_out in range(50): #算法总的循环次数50
        population = np_list
        step_in = 0
        while step_in < 50: #遗传操作最大迭代次数50
            step_in += 1
            dis_adp = []
            cum_loss = 0
            fitness = 0
            decoding_population = decode(population) #解码
            for i in range(num_total):
                individual = decoding_population[i:i+1]
                cum_loss = sess.run(cost,feed_dict={X:individual, y: batch_y, keep_prob:1}) #损失函数值的计算
                fitness = 1.0/cum_loss  #适应度计算，适应度为 1.0/损失函数值
                dis_adp.append(fitness)
            adp_max = max(dis_adp) #当前种群的最大适应度值
            adp_max_everygeneration.append(adp_max)
            adp_mean_tmp = sum(dis_adp)/len(dis_adp) #当前种群的平均适应度值
            adp_max_mean.append(adp_mean_tmp)
            print("最大适应度", adp_max)
            print("平均适应度", adp_mean_tmp)
            print(decoding_population)
            max_index = dis_adp.index(adp_max) #最大适应度值的索引位置
            population_max = population[max_index] #最大适应度值的个体

            population_save_DE = decoding_population
            opstructure_GA = population_save_DE[max_index]
            output_fitness_GA = adp_max
            if output_fitness_GA > 10: #阈值，若当前种群中最优个体的适应度值大于阈值，则保存
                outputfile.write("actual structure:" + str(x_input) + "\n")
                outputfile.write("only_GA predicted structure:" + str(opstructure_GA) + "\n")
                outputfile.write("only_GA Maximum fitness:" + str(output_fitness_GA) + "\n")
                print('actual structure:',x_input)
                print('total time:' + str(time.time()-start_time))
                outputfile.flush()
                break
            selection_population = selection(population, dis_adp) #选择操作
            crossover_population = crossover(selection_population) #交叉操作
            population = mutation(crossover_population) #变异操作

        tmp_structure_material = np.array([opstructure_GA[0:binary_len_material]])
        tmp_structure_thickness = np.array([opstructure_GA[binary_len_material:binary_len_material+numOfinput-1]])
        localsearch_loss, localsearch_structure = match_spectrum(tmp_structure_material, tmp_structure_thickness, num_layers, weights, biases, y_size, batch_y) #调用梯度优化算法
        if localsearch_loss < 0.1: #阈值，当梯度优化执行过后，若解的损失函数值小于阈值则保存
            optimum_codition = 1
            outputfile.write("actual structure:" + str(x_input) + "\n")
            outputfile.write("GA output structure " + " : "+ str(opstructure_GA) + "\n")
            outputfile.write("GA corresponding fitness " + " : "+ str(output_fitness_GA) + "\n")
            outputfile.write("GA and BP optimum structure:")
            for i in list(localsearch_structure[0]):
                outputfile.write(str(i) + ",")
            outputfile.write("\n")
            outputfile.write("GA and BP smalleset loss:" + str(localsearch_loss) + "\n")
            print('actual structure:',x_input)
            print('GA optimum structure:',opstructure_GA)
            print('GA Maximum fitness:',output_fitness_GA)
            print('total time:' + str(time.time()-start_time))
            outputfile.flush()
            break
        if current_localsearch_loss > localsearch_loss:
            current_localsearch_loss = localsearch_loss
            current_localsearch_structure = localsearch_structure
    if optimum_codition == 0: #如果算法没有找到一个小于阈值的解，则保存损失函数值最小的一个解
        outputfile.write("actual structure:" + str(x_input) + "\n")
        outputfile.write("GA and BP current optimum structure:")
        for i in list(current_localsearch_structure[0]):
            outputfile.write(str(i) + ",")
        outputfile.write("\n")
        outputfile.write("GA and BP current smalleset loss:" + str(current_localsearch_loss) + "\n")
        outputfile.flush()
        

       
outputfile.write("GA每次迭代最大的适应度:"+ str(adp_max_everygeneration) + "\n")
outputfile.write("GA每次迭代平均的适应度:" + str(adp_max_mean) + "\n")
outputfile.write("总时间:" + str(time.time()-start_time))
outputfile.flush()
outputfile.close()
sess.close()
