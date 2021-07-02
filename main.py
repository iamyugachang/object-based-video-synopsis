# ==== video_synopsis_main.py working flow ====
# - generate background
# - run deepsort & yolov4 to capture object frame 
# - merge data

#import section
from absl import app, flags, logging
from absl.flags import FLAGS
import time, os, json
import bg_extractor as bge
import video_synopsis as vs
import cv2


#flags definition
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './outputs/', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_bool('frame_cut',True, 'frame cut or not') #./outputs/frame_cut/...



def main(_argv):
    #background extraction
    print('Start background extraction') 
    bg_path, file_name = bge.bg_extr(FLAGS.video, './')
    print('background image:', bg_path)

    print('------------------------------------------')
    
    #tracking and merging
    print('Start detecting video synopsis')
    task1 = vs.VideoSynopsis(FLAGS.video,FLAGS.output, bg_path, file_name)
    task1.run()
    # task1.record_trajectory()
    return


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
    
    