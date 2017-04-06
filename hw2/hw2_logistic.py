#!/usr/bin/env python
# coding=utf-8
import math 
import numpy as np
import sys
import random, pickle

# read train data

train_xfile = open(sys.argv[3], "r", encoding ='utf-8')
next(train_xfile)
train_yfile = open(sys.argv[4], "r", encoding ='utf-8')
train_xdata = train_xfile.read().splitlines()

train_ydata = train_yfile.read().splitlines()

train_x = []
train_y = []

for i in range(len(train_xdata)):
    dat = train_xdata[i].split(',')[0::]
    
    dat_tmp = np.array(())
    for j in range(106):
        dat_tmp = np.hstack((dat_tmp, np.array(float(dat[j]))))
    if (i == 0):
        train_x = np.hstack((train_x,dat_tmp ))
    else:
        train_x = np.vstack((train_x, dat_tmp))
    train_y = np.hstack((train_y, np.array(float(train_ydata[i]))))

train_x = np.array(train_x)
train_y = np.array(train_y)

# Normalization
mean = np.sum(train_x, axis=0)/len(train_xdata)
std = (np.sum((train_x - mean)**2, axis=0)/len(train_xdata))**0.5
train_x = (train_x-mean)/std

print("Train")

# train

def error_function(w, b, result, data):
    offset = 0.0001
    z = (np.sum(data * w, axis=1) + b)
    f_wb = 1 / (1+ math.e ** (-z))
    cross_entropy = np.sum(result * np.log(f_wb+offset, np.array([math.e]*32561)) + \
                    (1-result) * np.log((1-f_wb+offset), np.array([math.e]*32561)))
    return -cross_entropy

# parameters
w = np.zeros((1, 106))
iterations = 5000
b = 0

# regularization
Lambda = 0

# Adadelta
grad_w = np.zeros((1, 106))
grad_b = 0
t_w = np.zeros((1, 106))
t_b = 0
T_w = np.zeros((1, 106))
T_b = 0
gamma = 0.9
epsilon = 10 ** -8

t = 1
while(True):
    z = np.sum(train_x * w, axis=1) + b
    f_wb = 1 / (1+ math.e ** (-z)) #sugmoid
    diff = train_y - f_wb 
    db = -1 * (diff.sum())
    dw = -1 * (np.sum(np.transpose(train_x) * diff, axis=1) - Lambda * w)

    # adadelta
    grad_w = gamma * grad_w + (1 - gamma) * (dw ** 2)
    grad_b = gamma * grad_b + (1 - gamma) * (db ** 2)
    t_w = -(((T_w + epsilon) ** 0.5) / ((grad_w + epsilon) ** 0.5))  * dw
    t_b = -(((T_b + epsilon) ** 0.5) / ((grad_b + epsilon) ** 0.5))  * db
    T_w = gamma * T_w + (1 - gamma) * (t_w ** 2)
    T_b = gamma * T_b + (1 - gamma) * (t_b ** 2)
    w += t_w
    b += t_b
    # debug
    if (t % 100 == 0):
        # print("#iter: ", t, "| Cross Entropy:", error_function(w, b, train_y, train_x) )
    if ( t > iterations):
        print ("Training is done.")
        break
    t += 1

# output the model
#model = open(sys.argv[5], "wb+")
#bind = (w, b, mean, std)
#pickle.dump(bind, model)
#model.close()

# read model
#model = open(sys.argv[5],"rb")
#w, b, mean, std = pickle.load(model)
#model.close()

# test
test_file = open(sys.argv[5], "r", encoding='utf-8')
next(test_file)
test_data = test_file.read().splitlines()

test = []
for i in range(len(test_data)):
    dat = test_data[i].split(',')[0::]
    tmp = np.array(())
    for j in range(106):
        tmp = np.hstack((tmp, np.array(float(dat[j]))))
    if (i==0):
        test = np.hstack((test,tmp))
    else:
        test = np.vstack((test, tmp))
test = np.array(test)
# test data normailization 
test = (test-mean) /std 

# test
res_z = np.sum(test*w, axis=1) + b
test_result = np.around(1/(1+math.e ** (-res_z)))

# output file
output_file = open(sys.argv[6],"w+")
output_file.write("id,label\n")

for i in range(len(test_data)):
    line = str(i+1)+","+str(int(test_result[i]))+"\n"
    output_file.write(line)
output_file.close()