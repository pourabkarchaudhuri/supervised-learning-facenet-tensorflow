#!/usr/bin/env python2
import cv2
import os
import sys
import numpy as np
sys.path.append('.')
import tensorflow as tf
import detect_face_detection
import time, shutil
import zerorpc

dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = dir_path + "\\images"

def main():
    c = zerorpc.Client()
    c.connect("tcp://127.0.0.1:4242")
    if os.path.exists("images"):
        shutil.rmtree("images")

    os.mkdir('images')
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    # filename_video = 'video.mp4' #Replace with Filename here of Video kept in /uploads Folder
    # file_name = os.getcwd() + "\\uploads\\" + filename_video
    video_capture = cv2.VideoCapture(1)
    # video_capture.set(3, 640)
    # video_capture.set(4, 480)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    minsize = 25 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    # colors = [tuple(255*np.random.rand(3)) for _ in range(10)]
    fps = 0
    frame_num = 0

    # sess = tf.Session()
    # sess = tf.Session() #Add GPU Module here
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face_detection.create_mtcnn(sess, None)

        while(True):
            start_time = time.time()
            # vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            ret, frame = video_capture.read()
            if not ret:
                break
            # Display the resulting frame
            img = frame[:,:,0:3]
            boxes, _ = detect_face_detection.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            print(boxes)
            for i in range(boxes.shape[0]):
                pt1 = (int(boxes[i][0]), int(boxes[i][1]))
                pt2 = (int(boxes[i][2]), int(boxes[i][3]))
                x = int(boxes[i][0]) - 40
                y = int(boxes[i][1]) - 40
                w = int(boxes[i][2]) + 40
                h = int(boxes[i][3]) + 40
                
                frame = cv2.rectangle(frame, (x,y), (w, h), color=(255, 255, 255))
                
                frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
                
                # if(float(boxes[i][4]) >= 0.90):
                    
                sub_faces = frame[y:h, x:w]
                # sub_faces = frame[p1, p2]
                path = UPLOAD_FOLDER + "\\" + "face_" + str(time.time()) + ".jpg"
                cv2.imwrite(path, sub_faces)
                result = c.classifyFile(path)
                # print(type(result[0]) == dict)
                if (len(result) != 0):
                    if (type(result[0]) == dict and len(result[0]['candidates']) != 0):
                        # result[0]['candidates']['name']
                        # print(result[0])
                        recognized_faces = result[0]
                        if (result[0]['candidates']['confidence']>0.96):
                            cv2.putText(img, recognized_faces['candidates']['name'] + " : " + str(round(recognized_faces['candidates']['confidence'], 2)), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
                        else:
                            cv2.putText(img, "Score too low", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(img, "No Faces Recognized", (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            end_time = time.time()
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time
            frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Video', frame)
            # p1 = int(boxes[0][2])
            # p2 = int(boxes[0][3])
            
            
            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
