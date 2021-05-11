import numpy as np
import cv2 
import os
import ntpath

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def bg_extr(video_path, result_path):
    # print(os.path.join(result_path, path_leaf(video_path).split('.')[-2]+'-background.jpg'))
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
    output_path = os.path.join(result_path, path_leaf(video_path).split('.')[-2]+'-bg.jpg')
    cv2.imwrite(output_path,background)
    cap.release()
    cv2.destroyAllWindows()
    return output_path
