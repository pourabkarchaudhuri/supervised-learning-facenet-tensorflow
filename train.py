import os, time
# import send_message

wait = 10
start = time.time()

os.system('python facenet/src/align_dataset_mtcnn.py  training/results training/aligned_dataset/ --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.50')
print("Waiting for "+str(wait)+" seconds and flushing GPU memory")
time.sleep(wait)
os.system('python facenet/src/classifier.py TRAIN training/aligned_dataset/ training/models/20170512-110547/20170512-110547.pb training/classifier/face_classifier.pkl --batch_size 1000 --min_nrof_images_per_class 20 --nrof_train_images_per_class 20')


time.sleep(wait)
print("Waiting for "+str(wait)+" seconds and flushing GPU memory")

# send_message.send_message()

end = time.time()
print("Execution Time : " + str((end - start)/60) + " mins.")

# os.system('python training/zerorpc_server.py')
# os.system('gnome-terminal -x training/zerorpc_client.py')