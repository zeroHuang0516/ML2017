import numpy as np
from sklearn.svm import LinearSVR as SVR
import sys
from sklearn.neighbors import NearestNeighbors

def get_eigenvalues(data):
    SAMPLE = 20 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals

# Train a linear SVR

npzfile = np.load('./train_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=1)
svr.fit(X, y)

# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict
testdata = np.load(sys.argv[1])
test_X = []
for i in range(200):
    #print(i)
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

with open(sys.argv[2], 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        print(f'{i},{np.log(d)}', file=f)