import cv2
import numpy as np
from numpy import *
import os
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

def readImg(dir):
    imgs = []
    imgList = ['A00.bmp','A01.bmp','A02.bmp','A03.bmp','A04.bmp','A05.bmp','A06.bmp','A07.bmp','A08.bmp','A09.bmp','B00.bmp','B01.bmp','B02.bmp','B03.bmp','B04.bmp','B05.bmp','B06.bmp','B07.bmp','B08.bmp','B09.bmp','C00.bmp','C01.bmp','C02.bmp','C03.bmp','C04.bmp','C05.bmp','C06.bmp','C07.bmp','C08.bmp','C09.bmp','D00.bmp','D01.bmp','D02.bmp','D03.bmp','D04.bmp','D05.bmp','D06.bmp','D07.bmp','D08.bmp','D09.bmp','E00.bmp','E01.bmp','E02.bmp','E03.bmp','E04.bmp','E05.bmp','E06.bmp','E07.bmp','E08.bmp','E09.bmp','F00.bmp','F01.bmp','F02.bmp','F03.bmp','F04.bmp','F05.bmp','F06.bmp','F07.bmp','F08.bmp','F09.bmp','G00.bmp','G01.bmp','G02.bmp','G03.bmp','G04.bmp','G05.bmp','G06.bmp','G07.bmp','G08.bmp','G09.bmp','H00.bmp','H01.bmp','H02.bmp','H03.bmp','H04.bmp','H05.bmp','H06.bmp','H07.bmp','H08.bmp','H09.bmp','I00.bmp','I01.bmp','I02.bmp','I03.bmp','I04.bmp','I05.bmp','I06.bmp','I07.bmp','I08.bmp','I09.bmp','J00.bmp','J01.bmp','J02.bmp','J03.bmp','J04.bmp','J05.bmp','J06.bmp','J07.bmp','J08.bmp','J09.bmp']
    for imgName in imgList:
        file_path = os.path.join(dir,imgName)
        img = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
        imgs.append(img.flatten())
    imgs = np.array(imgs)
    return imgs
    
def show_img(img_arr, h, w):
    img = []
    for i in range(h):
        img.append([])
        
    for i in range(h):
      for j in range(w):
          img[i].append(img_arr[0][i*h+j])
    img = np.array(img)
    return img
    

    
def judgeFace(judgeImg,faceVec,imgs_mean,diffImg,no):
    reduced = []
    for i in range(no):
        reduced.append((np.array(judgeImg-imgs_mean)*faceVec[i]))
    reduced = np.array(reduced)
    
    recover = imgs_mean
    
    for i in range(no):
        recover = recover +  (reduced[i]*faceVec[i])
        
    return recover  
        
    
imgs = readImg('./hp/face/')
imgs_mean = imgs.mean(axis=0,keepdims=True)
avgImg = show_img(imgs_mean, 64,64)
diffImg = imgs-imgs_mean
u, s, v = np.linalg.svd(diffImg)
#print(v.shape)

#for i in range(10):
#    eigenImg = show_img(v[i],64,64)
#    plt.figure()
#    eigen_fig = plt.gcf()
#    plt.imshow(eigenImg, cmap = cm.Greys_r)
#    #eigen_fig.savefig("./hp/eigenFace_"+str(i)+".png")
#    plt.imsave("./hp/eigenFace_"+str(i)+".png",eigenImg,cmap = cm.Greys_r)
#    plt.show()
#faceVec = [[],[],[],[],[]]   
#for i in range(5):
#    faceVec[i].append(v[i])
#faceVec = np.array(faceVec)

#for i in range(100):
#    recover = judgeFace(imgs[i],faceVec,imgs_mean,diffImg)
#    recover_img = show_img(recover, 64,64)
#    plt.imsave("./hp/recoveredFace_"+str(i)+".png",recover_img,cmap = cm.Greys_r)
smallest = 101;
smallest_k = 101;
for k in range(100):
    print("k= ",k)
    rmse=0
    faceVec = []
    for i in range(k+1):
        faceVec.append([])
    for i in range(k+1):
        faceVec[i].append(v[i])
    faceVec = np.array(faceVec)
    
    for i in range(100):
        recover = judgeFace(imgs[i],faceVec,imgs_mean,diffImg,k+1)
        rmse= rmse+np.sum(np.absolute(imgs[i]-recover))
    rmse = sqrt(rmse/4096/100)/256
    print(rmse)
    #if(rmse<0.01):
        #print("!!!!!!!!!!!!!!!!!")
    if (rmse<smallest):
        smallest = rmse
        smallest_k = k+1
print(smallest)
print(smallest_k)