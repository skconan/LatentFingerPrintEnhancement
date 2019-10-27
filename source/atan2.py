import cv2 as cv
import numpy as np
import math
def main():
    result = np.ones((200,200))
    x0,y0 = 100,100
    box = cv.boxPoints(((100,75),(1,50),np.rad2deg(0)))
    box = np.int0(box)
    cv.drawContours(result,[box],0,(0,0,255),1)
    for p in [[100,150],[100,50],[150,100],[50,100],[150,150],[50,150],[50,50],[150,50]]:
        result = np.ones((200,200))
     
        x,y = p
        print("p",p)
        angle = np.arctan2(-(y-y0),x-x0)
        print("angle")
        print(angle)
        print(np.rad2deg(angle))

        # box = cv.boxPoints(((100,75),(1,50),np.rad2deg(angle)))
        # box = np.int0(box)
        # cv.drawContours(result,[box],0,(0,0,255),1)
        x1,y1 = 100,100
        x2,y2 = x1 +50*np.cos(angle), y1 + 50*np.sin(angle)
        print("x2,y2")
        print(x2,y2)
        x2,y2 = int(math.ceil(x2)),int(math.ceil(y2))
        x1,y1 = int(math.ceil(x1)),int(math.ceil(y1))
        cv.line(result,(x1,y1),(x2,y2),(0,0,255),1)
        
        cv.imshow("result",result)
        cv.waitKey(-1)
if __name__ == "__main__":
    main()