#!/usr/bin/env python
# coding=utf-8
import math 
import numpy as np
import sys
import random, pickle

# read model
model = open('./model',"rb")
w, b, mean, std = pickle.load(model)
model.close()

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