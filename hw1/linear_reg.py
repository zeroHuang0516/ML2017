import csv
import numpy as np
import random
import sys

feature_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

# read the train data
train = np.genfromtxt(sys.argv[1],delimiter=',', dtype=None, skip_header=1)
row, col = np.shape(train)
data = []
for i in range(18):
	data.append([])

for i in range(row):
	if i%18 in feature_list:
		for val in train[i][3:]:
			if val == b'NR':
				data[i%18].append (float(0))
			else:
				data[i%18].append(float(val))
train = data


# some parameters
fold = 12
lenOfHour = 9
best_w = None
best_b = None
best_rmsErr = 100


# all the train data sets 
train_x = []
train_y = []

for m in range(12):
	for hs in range(471):
		x = []
		for t in feature_list:
			for hr in range(hs+9-lenOfHour, hs+9):
				x.append(data[t][480*m+hr])
		train_y.append(data[9][480*m+hs+9])
		train_x.append(x)

# shuffle train data
train_xy = zip(train_x,train_y)
train_xy = list(train_xy)
np.random.shuffle(train_xy)
train_x, train_y = zip(*train_xy)
train_x = np.array(train_x)
train_y = np.array(train_y)

# train
for idx in range(fold):
	x_train = []
	y_train = []
	x_valid = []
	y_valid = []
	for c in range(train_x.shape[0]):
		if idx*471 <= c and c<(idx+1)*471:
			x_valid.append(train_x[c])
			y_valid.append(train_y[c])
		else: 
			x_train.append(train_x[c])
			y_train.append(train_y[c])
	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_valid = np.array(x_valid)
	y_valid = np.array(y_valid)

	w = np.zeros(x_train.shape[1])
	b = 0

	# gradient 
	gradsq_w = np.zeros(x_train.shape[1])
	gradsq_b = 0.

	# learning rate for normal, adagrad, adam model
	eta_normal = 0.0000000001
	eta_ada = 1.
	eta_adam = 0.001

	# parameter for linear regularization
	Lambda = 10
	
	# parameters for adam models
	beta1 = 0.9
	beta2 = 0.999
	mt_w = 0
	vt_w = 0
	mt_b = 0
	vt_b = 0
	eps = 1e-8

	t = 0
	
	iters = 50000

	for i in range(iters):
		t+=1
		diff = y_train - (b + np.dot(x_train, w))
		x_train_t = x_train.T

		dw = -2 * np.dot(x_train_t, diff)
		db = -2 * np.sum(diff)


		# Regulation 
		dw += Lambda*2*w

		# normal model
#		w -= eta_normal*dw
#		b -= eta_normal*db

		# adagrad
		# gradsq_w += np.square(dw)
		# gradsq_b += np.square(db)
		# w -= eta_ada * dw /np.sqrt(gradsq_w)
		# b -= eta_ada * db /np.sqrt(gradsq_b)

		#adamgrad
		mt_w = beta1 * mt_w + (1-beta1) * dw
		vt_w = beta2 * vt_w + (1-beta2) * np.square(dw)
		mt_w_hat = mt_w / (1-np.power(beta1, t))
		vt_w_hat = vt_w / (1-np.power(beta2, t))

		mt_b = beta1 * mt_b + (1-beta1) *db
		vt_b = beta2 * vt_b + (1-beta2) * np.square(db)
		mt_b_hat = mt_b / (1-np.power(beta1, t))
		vt_b_hat = vt_b / (1-np.power(beta2, t))

		w -= (eta_adam * mt_w_hat) / (np.sqrt(vt_w_hat) + eps )
		b -= (eta_adam * mt_b_hat) / (np.sqrt(vt_b_hat) + eps )

	sumOfSqErr = np.sum( (y_train - ( b+np.dot(x_train, w) ) )**2)
	rmsErr_train = np.sqrt( sumOfSqErr / x_train.shape[0] )

	sumOfSqErr = np.sum( (y_valid - ( b+np.dot(x_valid ,w) ) )**2)
	rmsErr_valid = np.sqrt( sumOfSqErr / x_valid.shape[0] )
	
	rmsErr = (rmsErr_train + rmsErr_valid) / 2 
	if rmsErr < best_rmsErr:
		best_rmsErr = rmsErr
		best_w = w
		best_b = b

# debug checking




#read test data

test = np.genfromtxt(sys.argv[2], delimiter=',', dtype=None)
row, col = np.shape(test)
numOFTest = row / 18

testData = []
for i in range(18):
	testData.append([])

for i in range(row):
  if i%18 in feature_list:
    for j in range(2+9-lenOfHour, col):
      if test[i][j] == b'NR':
        testData[i%18].append(float(0))
      else:
        testData[i%18].append(float(test[i][j]))
test = testData



test_x = []
numOFTest = int(numOFTest)
for i in range(numOFTest):
	x = []
	for idx in feature_list:
		for hr in range(lenOfHour):
			x.append(test[idx][i*lenOfHour+hr])
	test_x.append(x)
test_x = np.array(test_x)


# predict
test_y = np.dot(test_x, best_w) + best_b
with open(sys.argv[3],'w+') as file:
	writer = csv.writer(file, delimiter=',')
	writer.writerow(('id', 'value'))
	for i in range(numOFTest):
		writer.writerow(('id_{0}'.format(i), test_y[i]))