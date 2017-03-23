import csv
import numpy as np
import random
import sys

feature_list = [0,4,5,6,7,8,9,12]

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
lenOfHour = 9
best_w = None
best_b = None
best_rmsErr = 100

# train data set
train_x = []
train_y = []

for m in range(12):
	for hs in range(471):
		x = []
		for t in feature_list:
			for hr in range(hs+9-lenOfHour, hs+9):
				if hr >= hs+9-2:
					x.append(data[t][480*m+hr])
					x.append(data[t][480*m+hr]**2)
				else:
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


x_train = train_x
y_train = train_y
x_valid = train_x
y_valid = train_y


# train
w = np.zeros(x_train.shape[1])
b = 0

# gradient
gradsq_w = np.zeros(x_train.shape[1])
gradsq_b = 0.

# learning rate 
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


while best_rmsErr > 5 and t < 500000:
	t += 1
	diff = y_train - (b+np.dot(x_train, w))
	x_train_t = x_train.T
	dw = -2 * np.dot(x_train_t, diff)
	db = -2 * np.sum(diff)


	# Regulation 
	dw += Lambda*2*w

	# Adagrad
	gradsq_w += np.square(dw)
	gradsq_b += np.square(db)
	w -= eta_ada * dw / np.sqrt(gradsq_w)
	b -= eta_ada * db / np.sqrt(gradsq_b)



sumOfSqErr = np.sum( ( y_valid - ( b + np.dot ( x_valid, w ) ) ) **2 )
rmsErr_valid = np.sqrt( sumOfSqErr / x_valid.shape[0] )

if rmsErr_valid < best_rmsErr:
	best_rmsErr = rmsErr_valid
	best_w = w
	best_b = b


# debug 




#read test data
test = np.genfromtxt(sys.argv[2], delimiter=',', dtype=None)
row, col = np.shape(test)
numOFTest = row/18

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
			if hr >= lenOfHour-2:
				x.append(test[idx][i*lenOfHour+hr])
				x.append(test[idx][i*lenOfHour+hr]**2)
			else:	
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