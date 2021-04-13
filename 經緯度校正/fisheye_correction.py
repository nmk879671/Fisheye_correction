import cv2
import numpy as np
import time
import math
import glob
import os
#基於經度魚眼校正
def undistort(img):
    m, n, k = img.shape[:3]
    print('m,n,k',m,n,k)
    result = np.zeros((m,n,k))
    R = max(m, n)/2
    Undistortion = []
    x = n/2
    y = m/2
    for u in range(m):
        for v in range(n):
            i = u
            j = round(math.sqrt(R ** 2 - (y - u) ** 2) * (v - x) / R + x)
            if (R ** 2 - (y - u) ** 2 < 0):
                continue
            result[u,v,0]=img[i,j,0]
            result[u,v,1]=img[i,j,1]
            result[u,v,2]=img[i,j,2]
    Undistortion = np.uint8(result)
    return Undistortion

if __name__ == "__main__":

    path = "under"
    for img in glob.glob(os.path.join(path,"*.jpg")):
        image = cv2.imread(img)
        output_name = path+"/fr_"+img.split('/')[-1]
        img_out = undistort(image)
        cv2.imwrite(output_name,img_out)
        print("fish eye distortion completely, save image to %s" % output_name)
