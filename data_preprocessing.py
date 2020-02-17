
import os
import cv2     # for capturing videos
import math   # for mathematical operations
import pandas as pd

import numpy as np    # for mathematical operations
#from skimage.transform import resize   # for resizing images

from glob import glob
from tqdm import tqdm
video_file_path = []
for path, subdirs, files in os.walk(os.path.realpath('/datahdd/student/RGB_for_opt/A040/')):
    for name in files:
        if name.endswith('.avi'):
            video_file_path.append(os.path.join(path, name))
    
for i in tqdm(range(len(video_file_path))):
    count = 0
    videoFile = video_file_path[i]
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    
    directory = videoFile[0:videoFile.rfind('/')]
    subdirectoryName = videoFile.split('/datahdd/student/RGB_for_opt/A040/')[1].split('/')[1]
    fileName = subdirectoryName.split(".")[0]
    os.mkdir(directory+"/"+fileName)

    while(True):
        # Extract images
        ret, frame = cap.read()
        # end of frames
        if not ret or count>180: 
            break
        
        filename_1 = directory+'/'+fileName+"/"+fileName+"_frame1%d%d.jpg" % ( i, count)
#         print(filename_1)
        count+=1
        cv2.imwrite(filename_1, frame)

    cap.release()
    #'/home/kamal/Desktop/server/rgb_data/NTU_data'