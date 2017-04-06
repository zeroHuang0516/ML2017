import csv
import numpy as np
import random
import sys

feature_list = list(range(106))
# read the train data
def read_train_data(x_datapath, y_datapath):
	train_x = []
	train_y = []
	x_data = np.genfromtxt(x_datapath, delimiter=',', skip_header =1)
	row, col = np.shape(x_data)

	for i in range(row):
            train_x.append([])

	for i in range(row):
		for c in range(col):
			if c in feature_list:
				train_x[i].append(float(x_data[i][c]))

	y_data = np.genfromtxt(y_datapath, delimiter=',',skip_header =1)
	# skip_header =1
	for row in y_data:
		train_y.append(float(row))
	train_x = np.array(train_x)
	train_y = np.array(train_y)

	return train_x, train_y


def read_test_data(test_datapath):
	test = np.genfromtxt(test_datapath, delimiter=',',skip_header =1)
	test_x = []
	for row in test:
		test_x.append(row[0:])

	test_x = np.array(test_x)
	return test_x

def sigmoid(x):
	return 1.0 / (1.0+np.exp(-x))

def cross_entropy(fw_b, y_n):
	ans = []
	for i in range(fw_b.shape[0]):
		if np.isclose(fw_b[i], 0.):
			ans.append( y_n[i] * np.log(fw_b[i]+1e-3) + (1-y_n[i]) * np.log( 1-fw_b[i] ) )
		elif np.isclose( 1-fw_b[i], 0.):
			ans.append( y_n[i] * np.log(fw_b[i]) + (1-y_n[i]) * np.log( 1-fw_b[i]+1e-3 ) )
		else:
			ans.append( y_n[i] * np.log(fw_b[i]) + (1-y_n[i]) * np.log( 1-fw_b[i] ) )
	return -np.array(ans)



def obtain_mean(X, Y):
    x_0,x_1 = [], []
    for i in range(Y.shape[0]):
        x_0.append(X[i]) if Y[i] == 0 else x_1.append(X[i])
    x_0, x_1 = np.array(x_0).T, np.array(x_1).T
    u_0, u_1 = [], []
    col = X.shape[1]
    for i in range(col):
        u_0.append(np.mean(x_0[i][:]))
        u_1.append(np.mean(x_1[i][:]))
    return np.array(u_0), np.array(u_1), x_0.shape[1], x_1.shape[1]
    
def obtain_cov(X):
    return np.cov(X, rowvar=False)

def write_model(u_0, u_1, cov, n_0, n_1):
    inv_cov = np.linalg.inv(cov)
    w = np.dot((u_0-u_1).T, inv_cov).T
    b = -0.5* np.dot(np.dot(u_0.T, inv_cov), u_0)+ \
          0.5* np.dot(np.dot(u_1.T, inv_cov), u_1)+ \
          np.log(n_0/n_1)
    #my_model = open(modelpath, 'w')
    #my_model.write(str(b))
    #my_model.write('\n')
    #for i in range(len(w)):
    #    if i < len(w) - 1:
    #        my_model.write(str(w[i]))
    #        my_model.write('\n')
    #    else:
    #        my_model.write(str(w[i]))
    return w, b


def read_model (modelpath):
	text = np.genfromtxt(modelpath, delimiter='\n')
	b = 0;
	w = []
	for i in range(len(text)):
		if(i == 0):
			b=text[i]
		else:
			w.append(text[i])
	return w, b

# predict
def predict(result_path, test_x, w, b):
	fwb_test = sigmoid(np.dot(test_x, w)+b)
	test_y = []
	for i in fwb_test:
		if np.less_equal(i, 0.5):
			test_y.append(0)
		else:
			test_y.append(1)

	with open(result_path, 'w') as file:
		writer = csv.writer(file, delimiter=',')
		writer.writerow(('id', 'label'))
		for i in range(len(test_y)):
			writer.writerow((i+1, test_y[i]))

if __name__ == '__main__':
    # train
    x, y = read_train_data(sys.argv[3], sys.argv[4])
    u_0, u_1, n_0, n_1 = obtain_mean(x,y)
    cov = obtain_cov(x)
    w, b = write_model(u_0,u_1,cov, n_0,n_1)
    # test
    test_x = read_test_data(sys.argv[5])
    #w, b = read_model(sys.argv[5]) 
    predict(sys.argv[6], test_x, w, b)
