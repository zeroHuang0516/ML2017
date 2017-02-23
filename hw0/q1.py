import sys
import numpy as np
mat1 = np.loadtxt(sys.argv[1], delimiter=',')
mat2 = np.loadtxt(sys.argv[2], delimiter=',')
mat = np.dot(mat1,mat2)
mat = sorted(mat)
f=open('ans_one.txt','w')
sys.stdout=f
for ele in mat:
  ele =(int)(ele)
  print(ele)
f.close()