import os
import glob
import time
import cv2
import numpy as np
import cPickle as cp

##-----------------------MAIN---------------------------------
##  Setup parameters
count = 0
stop = 100000
cDir = "images_train"
RGB_out = []

##  START
start_time = time.time()
in_files = np.asarray(glob.glob(os.path.join(cDir, "*.png")))
names = []
for img in in_files:
    name, ext = os.path.splitext(img)
    dirbuff, name = os.path.split(name)
    names.append("{:0>6}".format(name))
ordered = in_files[np.argsort(names)]
for imgIn in ordered:
    cImg = cv2.imread(imgIn)
    count += 1
    
    RGB_out.append(cImg)
    
    if count == stop:
        break
    elif (count % 5000) == 0:
        print count, "complete"
    elif count % 500 == 0:
        print '.',

RGB_out = np.swapaxes(np.asarray(RGB_out), 1, 3)
RGB_out = np.swapaxes(np.asarray(RGB_out), 2, 3)
mean = np.mean(RGB_out, axis=0)
stddev = np.std(RGB_out, axis=0)
normed = np.asarray((RGB_out - mean) / stddev, dtype=np.float32)
##  FINISH
print ("{0:.4f}".format(time.time() - start_time), " seconds for image processing")

cp.dump(normed, open("Train_mstd", 'wb'), 2)
