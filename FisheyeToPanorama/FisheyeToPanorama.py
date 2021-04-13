import cv2
import numpy as np
import time
import math
import glob
import os

def Filter_Tutoujing(src_img):   #凸透镜

    img_height=src_img.shape[0]
    img_width=src_img.shape[1]
    channel=src_img.shape[2]
    new_width = np.int(img_width/2 *3)
    #new_width = np.int(img_width* 2)
    new_height = np.int(img_height/2)
    
    new_img=np.zeros([new_height,new_width,channel],dtype=np.uint8)
    # x0=img_width/2
    # y0=img_width/2
    for i in range(0,new_width):
        for j in range(0,new_height):
            radius = new_height - j
            theta = math.pi * 2 / new_width * i * (-1)
            #x= x0 + j*math.cos(theta)
            #y= y0 + j*math.sin(theta)
            #alpha = math.pi *2*i/img_height/2

            # alpha = math.pi *2*i/img_height/2
            # x=x0 + ((j)* math.cos(alpha))
            # y=y0 + ((j)* math.sin(alpha))

            x = radius * math.cos(theta) + new_height #可調整theta角度
            y = new_height - (radius * math.sin(theta))

            if (x >= 0 and x < img_width and y >= 0 and y < img_height):
                src_x, src_y = x, y
                src_x_0 = int(src_x)
                src_y_0 = int(src_y)
                src_x_1 = min(src_x_0 + 1, img_width - 1)
                src_y_1 = min(src_y_0 + 1, img_height - 1)
                
                value0 = (src_x_1 - src_x) * src_img[src_y_0, src_x_0, :] + (src_x - src_x_0) * src_img[src_y_0, src_x_1, :] 
                value1 = (src_x_1 - src_x) * src_img[src_y_1, src_x_0, :] + (src_x - src_x_0) * src_img[src_y_1, src_x_1, :]
                
                # new_img[j,i,:] = src_img[np.int(y),np.int(x),:].astype('uint8')
                new_img[j,i,:] = ((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1 + 0.5).astype('uint8')
    return new_img





if __name__ == "__main__":
    path = "under"
    t =time.perf_counter()
    for img in glob.glob(os.path.join(path,"*.JPG")):
        image = cv2.imread(img)
        result_img =Filter_Tutoujing(image)
        output_name = "result/result_"+img.split('/')[-1]
        cv2.imwrite(output_name,result_img)
        print("fish eye distortion completely, save image to %s" % output_name)
        print(time.perf_counter()-t)
