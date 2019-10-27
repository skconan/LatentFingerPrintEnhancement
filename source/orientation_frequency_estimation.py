"""
    page 157

"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from constants import *
from utilities import *
from fft import *
import math
img = cv.imread(r"E:\KSIP\Database\NIST14_Binary\F0000002.jpg",0)
img = cv.blur(img,(3,3))
# img = cv.imread(r"E:\KSIP\Database\NIST27\Latent\G002L3U.bmp",0)
rows, cols = img.shape

# result = img.copy()
result_ff = np.ones((rows,cols),np.uint8)*255
result_of = np.ones((rows,cols),np.uint8)*255
# result_of = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
kernel = 64
step = 16
# kernel = 32
# step = 8
# bandpass = np.zeros((kernel,kernel),np.uint8)
cir1 = np.zeros((kernel,kernel),np.uint8)
cir2 = np.zeros((kernel,kernel),np.uint8)
cv.circle(cir1,(kernel//2,kernel//2),3,(255,255,255),-1)
cv.circle(cir2,(kernel//2,kernel//2),9,(255,255,255),-1)
bandpass = cir2 - cir1
cv.imshow("img",img)
cv.imshow("bandpass",bandpass)
for r in range(kernel//2,rows-kernel//2,step):
    for c in range(kernel//2,cols-kernel//2,step):
        roi = img[r-kernel//2:r+kernel//2, c-kernel//2:c+kernel//2].copy()
        spectrum = spatial2freq(roi)
        # dft = cv.dft(np.float32(roi),flags = cv.DFT_COMPLEX_OUTPUT)
        magnitude = get_magnitude(spectrum)
        # magnitude = normalize(magnitude)
        # magnitude = np.int16(magnitude)
        # print(magnitude.dtype)
        # print("max",magnitude.copy().max())
        # print("mean",magnitude.copy().mean())
        # print("dc",magnitude.copy()[kernel//2,kernel//2])
        # print("dft",dft[0,0])
        # print(magnitude[15,15])

        magnitude = magnitude - magnitude.max()
        magnitude = np.abs(magnitude)
        # magnitude = normalize(magnitude)
        # magnitude = np.clip(magnitude,0,magnitude.max())
        # magnitude = normalize(magnitude)
        # magnitude = normalize(to_logscale(magnitude))
        magnitude[kernel//2,kernel//2] = magnitude.max()
        # for i in [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]:
        #     magnitude[kernel//2+i[0],kernel//2+i[1]] = magnitude.max()
        # magnitude[magnitude==magnitude.max()] = 0
        magnitude = 255-normalize(magnitude)
        y,x = np.where(magnitude==magnitude.max())
        # print(x,y)
        magnitude = cv.bitwise_and(magnitude,magnitude,mask=bandpass)
        if len(x) >= 2:
            print("point max",x,y)
      
            # print(grad_x_mean)
            # p1 = x[0]-kernel//2,y[0]-kernel//2
            # p2 = x[1]-kernel//2,y[1]-kernel//2
            p1 = x[0],y[0]
            p2 = x[1],y[1]
            freq1 = np.sqrt((p1[0]-kernel//2)**2+(p1[1]-kernel//2)**2)
            freq2 = np.sqrt((p2[0]-kernel//2)**2+(p2[1]-kernel//2)**2)
            freq = np.mean([freq1,freq2])

            result_ff[r-step//2:r+step//2, c-step//2:c+step//2] = int(255*freq/64.)

            print("frequency",freq)
           
            point = np.ones((kernel,kernel),np.uint8)*255
            print(p1,p2)
            # if p2[1] < kernel//2:
            #     p = p2
            # else:
            #     p = p1
            
            # angle = np.arctan2(-(p[1]-kernel//2),p[0]-kernel//2) 
            angle = np.arctan2(-(p1[1]-p2[1]),p1[0]-p2[0]) 
            # if p[0] < kernel//2:
                # angle = np.arctan2(p2[1]-p1[1],p2[0]-p1[0]) #+ np.pi/2
                # angle = np.arctan2(p2[1]-kernel//2,p2[0]-kernel//2) 
            # else:
                # angle = np.arctan2(p1[1]-p2[1],p1[0]-p2[0]) #+ np.pi/2
                # angle = np.arctan2(p1[1]-kernel//2,p1[0]-kernel//2) 
            print("original angle",angle,np.rad2deg(angle))

            # shift 90 degrees 
            angle += np.pi/2
            
            # angle greater than pi
            if angle > np.pi:   
                # angle = -np.pi + angle - np.pi 
                angle = -2*np.pi + angle  
          
            print(angle)
            print("angle",np.rad2deg(angle))
            # x1,y1 = (c - step/2)*np.cos(angle), (r - step/2)*np.sin(angle)
            x1,y1 = c,r

            # quadrant
            if -np.pi/2 <= angle <= np.pi/2:
                x_sign = 1
            else:
                x_sign = -1
            
            if -np.pi <= angle <= 0:
                y_sign = 1
            else:
                y_sign = -1
      
            x2,y2 = x1 + x_sign * step / 2 * abs(np.cos(angle)), y1 + y_sign * step/2*abs(np.sin(angle))
            
            x2,y2 = int(math.ceil(x2)),int(math.ceil(y2))
            x1,y1 = int(math.ceil(x1)),int(math.ceil(y1))
            print(x1,y1,x2,y2)
            cv.line(result_of,(x1,y1),(x2,y2),(0,0,255),1)
            cv.line(point,(kernel//2,0),(kernel//2,kernel),(0,0,255),1)
            cv.line(point,(0,kernel//2),(kernel,kernel//2),(0,0,255),1)
            # box = cv.boxPoints(((c,r),(1,step),np.rad2deg(angle)))
            # box = np.int0(box)
            # cv.drawContours(result_of,[box],0,(0,0,255),1)
            # cv.line(result_of,(c+x[0],r+y[0]),(c+x[1],r+y[1]),(0,0,255),1)
            # cv.circle(point,(p[0],p[1]),1,(0,0,0),-1)



            cv.circle(point,(x[0],y[0]),1,(0,0,0),-1)
            cv.circle(point,(x[1],y[1]),1,(0,0,0),-1)
            imshow("point",point)
            cv.imshow('result_of',result_of)
            cv.imshow('result_ff',result_ff)
            imshow("roi",roi)

            mag_line = magnitude.copy()
            cv.line(mag_line,(kernel//2,0),(kernel//2,kernel),(255,255,255),1)
            cv.line(mag_line,(0,kernel//2),(kernel,kernel//2),(255,255,255),1)
            
            # imshow("magnitude",magnitude)
            # imshow("magnitude_line",mag_line)
            # imshow("magnitude_mapping",magnitude, mapping=True)
            cv.waitKey(1)
# cv.imshow('result_of',result_of)
# cv.imshow("x",sobelx)
# cv.imshow("y",sobely)
cv.imwrite("./of.jpg",result_of)
cv.imwrite("./ff.jpg",result_ff)
cv.waitKey(-1)