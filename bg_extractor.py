import numpy as np
import cv2 
import os


def bg_extr(video_path, result_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cnt = 0
    background  = np.zeros(frame.shape,np.float64)
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        background = background + frame/255.0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cnt += 1

    background /= cnt
    background *= 255
    cv2.imwrite(os.path.join(result_path, 'background.jpg'),background)
    cap.release()
    cv2.destroyAllWindows()
