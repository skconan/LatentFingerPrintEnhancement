import cv2 as cv
from utilities import *
from fft import *

frame = None
size = None
canvas = None
freq = None

def mouse_callback(event, cur_x, cur_y, flags, param):
    global frame, size, canvas, freq

    canvas = cv.cvtColor(frame.copy(), cv.COLOR_GRAY2BGR)

    rows, cols = frame.shape[:2]

    x_min = max(0, cur_x-size)
    y_min = max(0, cur_y-size)
    x_max = min(cols, cur_x+size)
    y_max = min(rows, cur_y+size)

    roi = frame[y_min:y_max, x_min:x_max]
    spectrum = spatial2freq(roi)
    magnitude = get_magnitude(spectrum)
    magnitude = magnitude_amplification(magnitude)
    magnitude = normalize(magnitude)

    cv.rectangle(canvas, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    y,x = np.where(magnitude==magnitude.max())

    if len(x) >= 2:
        print("point max",x,y)
        p1 = x[0],y[0]
        p2 = x[1],y[1]
        freq = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)/2.

    cv.putText(canvas, "Frequency = %.2f"%(freq), (10, 30),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    imshow("magnitude", magnitude)
    imshow("roi", roi)
    imshow("image", canvas)


def main():
    global frame, size, canvas
    cv.namedWindow("image")
    cv.setMouseCallback("image", mouse_callback)

    # size = kernel // 2
    size = 16
    # dir_sd27 = r"E:\OneDrive\KSIP\Database\NIST27\Latent"

    # img_path = dir_sd27 + "\G002L3U.bmp"
    dir_nist14 = r"E:\OneDrive\KSIP\Database\NIST14"
    img_path = dir_nist14 + r"\F0000001.png"
    frame = cv.imread(img_path, 0)
    canvas = cv.cvtColor(frame.copy(), cv.COLOR_GRAY2BGR)


    while True:
        cv.imshow("image", canvas)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break


if __name__ == "__main__":
    main()
