import cv2 as cv
import numpy as np

mask = []
rows = None
cols = None

direction = [
    # r,c
    [0,-1],
    [0,1],
    [-1,-1],
    [-1,0],
    [-1,1],
    [1,-1],
    [1,0],
    [1,1],
]

def quality(roi):
    return np.random.rand(1)

def feedback(img,r,c,step,ksize,color):
    global direction, rows, cols, mask
    print("feedback")
    quality_list = []
    mask.append([r,c])
    for d in direction:
        r_new = r + step*d[0]
        c_new = c + step*d[1]
        
        r_start = r_new-ksize//2
        r_end = r_new+ksize//2

        c_start = c_new-ksize//2
        c_end = c_new+ksize//2

        if r_start < 0 or r_end > rows or c_start < 0 or c_end > cols:
            quality_list.append(-1)
        elif [r_new, c_new] in mask:
            quality_list.append(-1)
        else:            
            roi = img[r_start:r_end, c_start:c_end]
            q = quality(roi)
            quality_list.append(q)
        
    quality_list = np.float32(quality_list)
    maximum = quality_list.max()
    if maximum == -1:
        return None
    index = np.where(quality_list==maximum)
    index = index[0][0]
    r_new = r + step*direction[index][0]
    c_new = c + step*direction[index][1]

    cv.circle(img,(c,r),2,color,-1)
    cv.rectangle(img, (c-ksize//2,r-ksize//2), (c+ksize//2,r+ksize//2), color, 2)
    cv.rectangle(img, (c-step//2,r-step//2), (c+step//2,r+step//2), color, -1)
    
    cv.imshow("img",img)
    cv.waitKey(-1)
    return feedback(img, r_new, c_new, step, ksize, color)

def main():
    global rows, cols
    img = cv.imread(r"E:\KSIP\LatentFingerPrintEnhancement\images\F0000032.png")
    rows, cols, _ = img.shape
    step = 16
    ksize = 64
    point = []
    for r in range(ksize//2, rows - ksize//2, step):
       for c in range(ksize//2, cols - ksize//2, step):
           point.append([r,c])
        #    cv.circle(img,(c,r),2,(100),-1)
        #    cv.rectangle(img, (c-ksize//2,r-ksize//2), (c+ksize//2,r+ksize//2), (100), 1)
        #    cv.imshow("img",img)
        #    cv.waitKey(-1)
    while len(mask) < len(point):
        print("while")
        color = (np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255))
        print(len(point))
        start = np.random.randint(0,len(point)-1)
        r_start, c_start = point[start]
        feedback(img,r_start,c_start,step,ksize,color)
    print("end")
if __name__ == "__main__":
    main()