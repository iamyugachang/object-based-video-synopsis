#ignore warning
import warnings  
warnings.filterwarnings("ignore")  
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0:1], 'GPU')
# print(physical_devices)
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import json
import shutil
from tqdm import tqdm

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')


class VideoSynopsis():
    def __init__(self, input_path, output_path, bg_path, file_name) -> None:
        self.framework = 'tf'
        self.weights = './checkpoints/yolov4-416'
        self.size = 416
        # self.tiny = False
        # self.model = 'yolov4'
        self.video = input_path
        self.output = output_path
        self.output_format = 'XVID'
        self.iou = 0.45
        self.score = 0.50
        self.dont_show = False
        self.info = False
        self.count = False
        self.bg_path = bg_path
        self.file_name = file_name
        
    
    def run(self):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        frame_dict_bin = {} #ex: frame_dict =  {track_id: [frame1, frame2, ...]}
        frame_dict_rgb = {}
        position_dict = {} #ex: position_dict = {track_id: [bbox1, bbox2, bbo3...]}
        enter_time_dict = {} #ex: enter_time_dict = {track_id: time}
        tube_mask_dict = {} #ex: tube_mask_dict = {trak_id:tube}
        enter_queue = [] #ex: enter_queue = [id1, id2, id3]
        
        #create folder for frame_cut
        if FLAGS.frame_cut:
            self.path_frame_cut = os.path.join(self.output, 'frame_cut', self.file_name)
            if os.path.exists(self.path_frame_cut):
                shutil.rmtree(self.path_frame_cut, ignore_errors=True)
                os.mkdir(self.path_frame_cut)
            else:
                os.mkdir(self.path_frame_cut)
            

        
        # initialize deep sort
        print('Establishing object tracker...')
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        print('Establishing object detector...')
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = self.size
        video_path = self.video

        # load tflite model if flag is set
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
        
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        original_fps = vid.get(cv2.CAP_PROP_FPS)
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        #capture background
        back = cv2.imread(self.bg_path)
        back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        back_blur = cv2.GaussianBlur(back_gray, (21, 21), 0)

        backSub = cv2.createBackgroundSubtractorMOG2()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # while video is running
        print('Start detecting videos!')
        
        # while True:
        # tmp_count = 0
        for frame_num in tqdm(range(length)):
        # for frame_num in tqdm(range(100)):
            return_value, frame = vid.read()
            
            if return_value:
                frame_original = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Finished!')
                break
            
            # print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if self.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou,
                score_threshold=self.score
            )
            
            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            # allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['person', 'car']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            # if FLAGS.count:
            #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            #     print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            #background abstraction
            frame_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
            difference = cv2.absdiff(gray_blur, back_blur)
            thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=15) #original iterations = 15

            # background abstraction MOG
            # frame_tmp = cv2.GaussianBlur(frame_original, (21, 21), 0)
            # frame_tmp = cv2.morphologyEx(frame_tmp, cv2.MORPH_CLOSE, kernel, iterations=2)
            # thresh = backSub.apply(frame_tmp)
            # thresh = cv2.dilate(thresh, None, iterations=20)

            # update tracks
            new_frame = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) 
            new_frame_rgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) & frame_original
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                #crop ROI
                canvas = np.zeros(frame_original.shape,dtype=np.uint8)
                canvas[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255

                # expansion:
                y_size = abs(bbox[1] - bbox[3])
                x_size = abs(bbox[0] - bbox[2])
                new_ymin = int(bbox[1]-y_size*0.2)
                new_ymax = int(bbox[3]+y_size*0.2)
                new_xmin = int(bbox[0]-x_size*0.2)
                new_xmax = int(bbox[2]+x_size*0.2)
                # canvas = np.zeros(frame_original.shape,dtype=np.uint8)
                # canvas[new_ymin:new_ymax, new_xmin:new_xmax] = 255
                
                bin_cropped_frame = cv2.bitwise_and(canvas, new_frame)
                rgb_cropped_frame = cv2.bitwise_and(canvas, new_frame_rgb)
                # cv2.imwrite("./outputs/tmp/"+str(tmp_count)+".png",rgb_cropped_frame)
                # tmp_count+=1
                if track.track_id not in enter_queue:
                    enter_queue.append(track.track_id)

                if track.track_id not in enter_time_dict:
                    enter_time_dict[track.track_id] = round(frame_num/original_fps, 2)

                if track.track_id in position_dict:
                    position_dict[track.track_id].append(bbox)
                else:
                    position_dict[track.track_id] = [bbox]

                # if track.track_id in frame_dict_bin:
                #     frame_dict_bin[track.track_id].append(bin_cropped_frame)
                # else:
                #     frame_dict_bin[track.track_id] = [bin_cropped_frame]

                if track.track_id in frame_dict_rgb:
                    frame_dict_rgb[track.track_id].append(rgb_cropped_frame)
                else:
                    frame_dict_rgb[track.track_id] = [rgb_cropped_frame]

                _ , tube_mask = cv2.threshold(rgb_cropped_frame, 10, 255, cv2.THRESH_BINARY)
                if track.track_id in tube_mask_dict:
                    tube_mask_dict[track.track_id] = cv2.add(tube_mask_dict[track.track_id], tube_mask)
                else:
                    tube_mask_dict[track.track_id] = tube_mask
                
                # Generate mask of objects
                # ret, tmp = cv2.threshold(rgb_cropped_frame, 10, 255, cv2.THRESH_BINARY)
                # cv2.imwrite('./outputs/process/'+str(track.track_id)+'_'+str(frame_num)+'.png',tmp)
                
            # if enable info flag then print details about each track
                if self.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            # calculate frames per second of running detections
            # fps = 1.0 / (time.time() - start_time)
            # print("FPS: %.2f" % fps)
            # result = np.asarray(frame)
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # if not FLAGS.dont_show:
            #     cv2.imshow("Output Video", result)
            
            # # if output flag is set, save video file
            # if FLAGS.output:
            #     out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

        #Generate videos
        out = None
        # get video ready to save locally if flag is set
        if self.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(os.path.join(self.output, self.file_name+'_synopsis.avi'), codec, fps, (width, height))
        count = 0
        print('Start outputting synopsis video')
        ###UNDER CONSTRUCTION
        pipeline = [] #pipeline = [ [id1, id2, id3], [id4, id5]....[], [], ....]
        #Step 1: select candidate tube in FIFO order'
        for id in tube_mask_dict:
            cv2.imwrite("./outputs/tube/"+str(id)+".png", tube_mask_dict[id])
        while True:
            selected_id = []
            tmp_queue = []
            while len(enter_queue) != 0:
                collision = False
                current = enter_queue.pop(0)
                if len(selected_id) == 0:
                    selected_id = [current]
                else:
                    for id in selected_id:
                        # intensity = cv2.bitwise_and(tube_mask_dict[current], tube_mask_dict[id]).sum()
                        enter_time_gap = abs(enter_time_dict[current] - enter_time_dict[id])
                        area_current = np.count_nonzero((tube_mask_dict[current] == [255,255,255]).all(axis=2))
                        area_compare = np.count_nonzero((tube_mask_dict[id] == [255,255,255]).all(axis=2))
                        overlap_tmp = cv2.bitwise_and(tube_mask_dict[current], tube_mask_dict[id])
                        area_overlap = np.count_nonzero((overlap_tmp == [255,255,255]).all(axis=2))
                        iou = area_overlap / (area_current + area_compare - area_overlap + 0.001) 
                        if enter_time_gap > 1 and iou > 0.2:
                            #collision happend
                            collision = True
                            # print("overlap, current, compare, iou = ", area_overlap, area_current, area_compare, iou)
                            break
                    if collision:
                        tmp_queue.append(current)
                    else:
                        selected_id.append(current)
                    collision = False
            enter_queue = tmp_queue.copy()
            pipeline.append(selected_id)
            print("selected queue =", selected_id)
            if len(enter_queue) == 0:
                break
        #Step 2: output objects according to the pipeline order
        for order in pipeline:
            while True:
                base_bin = np.zeros(frame_original.shape,dtype=np.uint8)
                base_rgb = np.zeros(frame_original.shape,dtype=np.uint8)
                count_zero = 0
                frame_map = []
                for id in order:
                    if len(frame_dict_rgb[id])!=0:
                        frame_rgb = frame_dict_rgb[id].pop(0)
                        bbox = position_dict[id].pop(0)
                        img1 = frame_rgb
                        img2 = base_rgb
                        _, img1_mask = cv2.threshold(frame_rgb, 10, 255, cv2.THRESH_BINARY)
                        _, img2_mask = cv2.threshold(base_rgb, 10, 255, cv2.THRESH_BINARY)
                        _, overlap_mask = cv2.threshold(cv2.bitwise_and(img1_mask, img2_mask), 10, 255, cv2.THRESH_BINARY)
                        overlap_mask_inv = cv2.bitwise_not(overlap_mask)
                        blended_part = cv2.addWeighted(img1 & overlap_mask, 0.5, img2&overlap_mask, 0.5, 0)
                        base_rgb = ((img1 + img2) & overlap_mask_inv) + blended_part
                        # base_rgb = cv2.add(base_rgb, frame_rgb) #where cause overlap
                        # ret, frame_bin = cv2.threshold(frame_rgb, 10, 255, cv2.THRESH_BINARY)
                        frame_bin = img1_mask
                        base_bin = cv2.add(base_bin, frame_bin)
                        cv2.putText(base_rgb, str(enter_time_dict[id])+" (s)",(int(bbox[0]), int(bbox[1]+70)),0, 0.75, (255,255,255),2)
                    else:
                        count_zero+=1
                base_bin_inv = cv2.bitwise_not(base_bin)
                a = cv2.bitwise_and(base_bin_inv, back)
                res = cv2.add(a, base_rgb)
                
                if FLAGS.frame_cut:
                    cv2.imwrite(os.path.join(self.path_frame_cut, str(count)+'.jpg'), res)
                out.write(res)
                count+=1
                if count_zero >= len(order):
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
        out.release()
        ###UNDER CONSTRUCTION
        '''
        while True:
            base_bin = np.zeros(frame_original.shape,dtype=np.uint8)
            base_rgb = np.zeros(frame_original.shape,dtype=np.uint8)
            count_zero = 0
            frame_map = []
            # for id in frame_dict_bin:
            #     if len(frame_dict_bin[id])!=0:
            #         base_bin = cv2.add(base_bin, frame_dict_bin[id].pop(0))
            # for id in frame_dict_rgb:
            #     cv2.imwrite('./outputs/tube_mask/'+str(count)+'.png', tube_mask_dict[id])
            #     count+=1
            # return 
            for id in frame_dict_rgb:
                if len(frame_dict_rgb[id])!=0:
                    frame_rgb = frame_dict_rgb[id].pop(0)
                    bbox = position_dict[id].pop(0)
                    # bbox = [int(pos) for pos in bbox]
                    # center = ( int(round((bbox[0]+bbox[2])/2)), int(round((bbox[1]+bbox[3])/2)) ) #(x,y)
                    
                    # if base_rgb[ bbox[1]:bbox[3], bbox[0]:bbox[2]].sum() == 0:
                    #     base_rgb = cv2.add(base_rgb, frame_rgb)
                    # else:
                    #     base_rgb = cv2.addWeighted(base_rgb, 0.5, frame_rgb, 0.5, 0)
                    base_rgb = cv2.add(base_rgb, frame_rgb)
                    ret, frame_bin = cv2.threshold(frame_rgb, 10, 255, cv2.THRESH_BINARY)
                    base_bin = cv2.add(base_bin, frame_bin)
                    cv2.putText(base_rgb, str(enter_time_dict[id])+" (s)",(int(bbox[0]), int(bbox[1]+70)),0, 0.75, (255,255,255),2)
                else:
                    count_zero+=1
            
            base_bin_inv = cv2.bitwise_not(base_bin)
            a = cv2.bitwise_and(base_bin_inv, back)
            res = cv2.add(a, base_rgb)
            
            if FLAGS.frame_cut:
                cv2.imwrite(os.path.join(self.path_frame_cut, str(count)+'.jpg'), res)
            out.write(res)
            count+=1
            if count_zero >= len(frame_dict_rgb):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
        out.release()
        '''
            
        print('Finished!')
        return

    def record_trajectory(self):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        frame_dict_bin = {} #ex: frame_dict =  {track_id: [frame1, frame2, ...]}
        frame_dict_rgb = {}
        position_dict = {} #ex: position_dict = {track_id: [bbox1, bbox2, bbo3...]}
        enter_time_dict = {} #ex: enter_time_dict = {track_id: time}
        
        #create folder for frame_cut
        if FLAGS.frame_cut:
            self.path_frame_cut = os.path.join(self.output, 'frame_cut', self.file_name)
            if os.path.exists(self.path_frame_cut):
                shutil.rmtree(self.path_frame_cut, ignore_errors=True)
                os.mkdir(self.path_frame_cut)
            else:
                os.mkdir(self.path_frame_cut)
            

        
        # initialize deep sort
        print('Establishing object tracker...')
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        print('Establishing object detector...')
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = self.size
        video_path = self.video

        # load tflite model if flag is set
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
        
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        original_fps = vid.get(cv2.CAP_PROP_FPS)

        #capture background
        back = cv2.imread(self.bg_path)
        back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        back_blur = cv2.GaussianBlur(back_gray, (21, 21), 0)

        frame_num = 0
        id_color = {} #ex: {id:color}
        frame_data = []
        #ex: frame_data = [ {id1:pos1, id2:pos2...},...,...]
        # while video is running
        print('Start detecting videos!')
        # while True:
        for c in tqdm(range(original_fps)):
            return_value, frame = vid.read()
            
            if return_value:
                frame_original = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Finished!')
                break
            frame_num +=1
            # print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if self.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou,
                score_threshold=self.score
            )
            
            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            # allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['person', 'car']
            # allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            # if FLAGS.count:
            #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            #     print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            #background abstraction
            frame_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
            difference = cv2.absdiff(gray_blur, back_blur)
            thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # update tracks
            new_frame = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) 
            new_frame_rgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) & frame_original
            pos_per_frame = {}
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                #random choose color
                if track.track_id not in id_color:
                    id_color[track.track_id] = list(np.random.random(size=3)*256)
                #store data in each frame
                pos_per_frame[track.track_id] = (round((int(bbox[0])+int(bbox[2]))/2), round((int(bbox[1])+int(bbox[3]))/2))
                
            # if enable info flag then print details about each track
                if self.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            # calculate frames per second of running detections
            # fps = 1.0 / (time.time() - start_time)
            # print("FPS: %.2f" % fps)
            # result = np.asarray(frame)
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # if not FLAGS.dont_show:
            #     cv2.imshow("Output Video", result)
            
            # # if output flag is set, save video file
            # if FLAGS.output:
            #     out.write(result)
            #store
            frame_data.append(pos_per_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

        #Generate videos
        out = None
        # get video ready to save locally if flag is set
        if self.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(os.path.join(self.output, self.file_name+'_trajectory.avi'), codec, fps, (width, height))
        count = 0
        print('Start outputting trajectory video')
        canvas = back
        for frame_id in range(len(frame_data)):
            
            current_frame_data = frame_data[frame_id]
            # print('processing frame', frame_id, 'with', len(current_frame_data), 'objects')
            # print(current_frame_data)
            for id in current_frame_data:
                # print(current_frame_data[id], id_color[id])
                cv2.circle(canvas, current_frame_data[id], radius=5, color=id_color[id], thickness=-1)
            if FLAGS.frame_cut:
                cv2.imwrite(os.path.join(self.path_frame_cut, str(count)+'.jpg'), canvas)
            out.write(canvas)
            cv2.imshow('new', canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
        out.release()
            
        # while True:
        #     base_bin = np.zeros(frame_original.shape,dtype=np.uint8)
        #     base_rgb = np.zeros(frame_original.shape,dtype=np.uint8)
        #     count_zero = 0
        #     for id in frame_dict_bin:
        #         if len(frame_dict_bin[id])!=0:
        #             base_bin = cv2.add(base_bin, frame_dict_bin[id].pop(0))
        #     for id in frame_dict_rgb:
        #         if len(frame_dict_rgb[id])!=0:
        #             base_rgb = cv2.add(base_rgb, frame_dict_rgb[id].pop(0))
        #             bbox = position_dict[id].pop(0)
        #             cv2.putText(base_rgb, str(enter_time_dict[id])+" (s)",(int(bbox[0]), int(bbox[1]+70)),0, 0.75, (255,255,255),2)
        #         else:
        #             count_zero+=1
            
        #     base_bin_inv = cv2.bitwise_not(base_bin)
        #     a = cv2.bitwise_and(base_bin_inv, back)
        #     res = cv2.add(a, base_rgb)
            
        #     if FLAGS.frame_cut:
        #         cv2.imwrite(os.path.join(self.path_frame_cut, str(count)+'.jpg'), res)
        #     out.write(res)
        #     count+=1
        #     if count_zero >= len(frame_dict_rgb):
        #         break
        #     if cv2.waitKey(1) & 0xFF == ord('q'): break
        # cv2.destroyAllWindows()
        # out.release()

            
        print('Finished!')
        return
    
    def run_generate_sample_video(self):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        frame_dict_bin = {} #ex: frame_dict =  {track_id: [frame1, frame2, ...]}
        frame_dict_rgb = {}
        position_dict = {} #ex: position_dict = {track_id: [bbox1, bbox2, bbo3...]}
        enter_time_dict = {} #ex: enter_time_dict = {track_id: time}
        
        #create folder for frame_cut
        if FLAGS.frame_cut:
            self.path_frame_cut = os.path.join(self.output, 'frame_cut', self.file_name)
            if os.path.exists(self.path_frame_cut):
                shutil.rmtree(self.path_frame_cut, ignore_errors=True)
                os.mkdir(self.path_frame_cut)
            else:
                os.mkdir(self.path_frame_cut)
            

        
        # initialize deep sort
        print('Establishing object tracker...')
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        # load configuration for object detector
        print('Establishing object detector...')
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = self.size
        video_path = self.video

        # load tflite model if flag is set
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']
        
        # begin video capture
        try:
            vid = cv2.VideoCapture(int(video_path))
        except:
            vid = cv2.VideoCapture(video_path)

        original_fps = vid.get(cv2.CAP_PROP_FPS)

        #capture background
        back = cv2.imread(self.bg_path)
        back_gray = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
        back_blur = cv2.GaussianBlur(back_gray, (21, 21), 0)

        frame_num = 0

        # while video is running
        print('Start detecting videos!')
        while True:
            return_value, frame = vid.read()
            
            if return_value:
                frame_original = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Finished!')
                break
            frame_num +=1
            # print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if self.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou,
                score_threshold=self.score
            )
            
            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            # allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to customize tracker for only people)
            allowed_classes = ['person', 'car']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)
            # if FLAGS.count:
            #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            #     print("Objects being tracked: {}".format(count))
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            #background abstraction
            frame_gray = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
            difference = cv2.absdiff(gray_blur, back_blur)
            thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            # update tracks
            new_frame = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) 
            new_frame_rgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB) & frame_original
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                #crop ROI
                canvas = np.zeros(frame_original.shape,dtype=np.uint8)
                canvas[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
                
                bin_cropped_frame = canvas & new_frame
                rgb_cropped_frame = canvas & new_frame_rgb
                if track.track_id not in enter_time_dict:
                    enter_time_dict[track.track_id] = round(frame_num/original_fps, 2)

                if track.track_id in position_dict:
                    position_dict[track.track_id].append(bbox)
                else:
                    position_dict[track.track_id] = [bbox]

                if track.track_id in frame_dict_bin:
                    frame_dict_bin[track.track_id].append(bin_cropped_frame)
                else:
                    frame_dict_bin[track.track_id] = [bin_cropped_frame]

                if track.track_id in frame_dict_rgb:
                    frame_dict_rgb[track.track_id].append(rgb_cropped_frame)
                else:
                    frame_dict_rgb[track.track_id] = [rgb_cropped_frame]
                
            # if enable info flag then print details about each track
                if self.info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            # calculate frames per second of running detections
            # fps = 1.0 / (time.time() - start_time)
            # print("FPS: %.2f" % fps)
            # result = np.asarray(frame)
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # if not FLAGS.dont_show:
            #     cv2.imshow("Output Video", result)
            
            # # if output flag is set, save video file
            # if FLAGS.output:
            #     out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()

        #Generate videos
        out = None
        # get video ready to save locally if flag is set
        if self.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(os.path.join(self.output, self.file_name+'_synopsis.avi'), codec, fps, (width, height))
        count = 0
        print('Start outputting synopsis video')
        while True:
            base_bin = np.zeros(frame_original.shape,dtype=np.uint8)
            base_rgb = np.zeros(frame_original.shape,dtype=np.uint8)
            count_zero = 0
            for id in frame_dict_bin:
                if len(frame_dict_bin[id])!=0:
                    base_bin = cv2.add(base_bin, frame_dict_bin[id].pop(0))
            for id in frame_dict_rgb:
                if len(frame_dict_rgb[id])!=0:
                    base_rgb = cv2.add(base_rgb, frame_dict_rgb[id].pop(0))
                    bbox = position_dict[id].pop(0)
                    cv2.putText(base_rgb, str(enter_time_dict[id])+" (s)",(int(bbox[0]), int(bbox[1]+70)),0, 0.75, (255,255,255),2)
                else:
                    count_zero+=1
            
            base_bin_inv = cv2.bitwise_not(base_bin)
            a = cv2.bitwise_and(base_bin_inv, back)
            res = cv2.add(a, base_rgb)
            
            if FLAGS.frame_cut:
                cv2.imwrite(os.path.join(self.path_frame_cut, str(count)+'.jpg'), res)
            out.write(res)
            count+=1
            if count_zero >= len(frame_dict_rgb):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
        out.release()

            
        print('Finished!')
        return
