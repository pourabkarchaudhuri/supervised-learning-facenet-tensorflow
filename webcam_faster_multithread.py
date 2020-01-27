
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread, Lock
# import cv2


import tensorflow as tf
import argparse
import training.facenet
import os
import sys
import math
import pickle
import training.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC 
import detect_face_detection
import time

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help = 'Path of the video you want to test on.', default = 0)
    args = parser.parse_args()
    
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'training/classifier/face_classifier.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'training/models/20170512-110547/20170512-110547.pb'
    fps = 0
    frame_num = 0
    
    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")
            
            
    with tf.Graph().as_default():
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            
            # Load the model
            print('Loading feature extraction model')
            
            training.facenet.load_model(FACENET_MODEL_PATH)
            
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            pnet, rnet, onet = detect_face_detection.create_mtcnn(sess, os.path.join(os.getcwd()))
            
            
            people_detected = set()
            person_detected = collections.Counter()
            
            cap = cv2.VideoCapture(VIDEO_PATH)
            
            while(cap.isOpened()):
                start_time = time.time()
                ret, frame = cap.read()
                
                bounding_boxes, _ = detect_face_detection.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                
                faces_found = bounding_boxes.shape[0]
                
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        # print(det)
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                            scaled = training.facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            # print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                        
                            
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 255, 255), 1)
                            # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3] + 20), (255, 255, 255), cv2.FILLED)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20  
                            
                            if best_class_probabilities > 0.92:
                                name = class_names[best_class_indices[0]]
                            else:
                                name = "Unauthorized"
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.50, (255, 255, 255), thickness=1)
                            
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.50, (0, 0, 0), thickness=1)
                            person_detected[best_name] += 1
                except:
                    pass
                
                end_time = time.time()
                fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
                start_time = end_time
                frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
                cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow('Face Recognition',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
main()

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        # self.thread.join()
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()

if __name__ == "__main__" :
    vs = WebcamVideoStream().start()
    while True :
        frame = vs.read()
        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()