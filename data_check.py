import os
import numpy as np
import cv2
import random

image_path = '/datahdd/student/RGB_for_opt/'

size = 112

# There are 40 actions or labels
# for training - 70 videos inside each action
# for testing - 30 videos inside each action

# we take 16 consecutive images to define a sequence
seq_length = 16
actions = os.listdir(image_path)
actions.sort(key=str.lower)

for action in actions:
    for job in ['train_data','test_data']:
        samples = [ name for name in os.listdir(image_path+action+'/'+job) if os.path.isdir(os.path.join(image_path+action+'/'+job, name)) ]
        #samples = os.listdir(image_path+action+'/'+job)
        samples.sort(key=str.lower)#
        if(len(samples) != 30 and len(samples) != 70):
            print('No. of samples in '+action+' in '+job+' : ', len(samples))
        k = 0
        kk = 0
        for sample in samples:
            if job == 'train_data':
                thedir = image_path+action+'/'+job+'/'+sample
                images = [ name for name in os.listdir(thedir) if not os.path.isdir(os.path.join(thedir, name)) ]
                #images = os.listdir(image_path+action+'/'+job+'/'+sample)
                images.sort(key=str.lower)
                if(len(images) < seq_length):
                    print('No. of images in '+action+' in '+job+' in '+sample+' : ', len(images))
                frames = [i for i in range(len(images)-(seq_length-1))]
                #print('No. of frames in '+action+' in '+job+' in '+sample+' : ', len(frames))
                start_frame = random.sample(frames,1)
                #print('No. of start frames in '+action+' in '+job+' in '+sample+' : ', len(start_frame))
                for frame in start_frame:
                    for j in range(seq_length):
                        
                        k += 1
                
                        
            else:
                thedir = image_path+action+'/'+job+'/'+sample
                images = [ name for name in os.listdir(thedir) if not os.path.isdir(os.path.join(thedir, name)) ]
                #images = os.listdir(image_path+action+'/'+job+'/'+sample)
                images.sort(key=str.lower)
                if(len(images) < seq_length):
                    print('No. of images in '+action+' in '+job+' in '+sample+' : ', len(images))
                frames = [i for i in range(len(images) - (seq_length-1))]
                #print('No. of frames in '+action+' in '+job+' in '+sample+' : ', len(frames))
                start_frame = random.sample(frames, 1)
                for frame in start_frame:
                    for j in range(seq_length):
                        kk += 1
        #if(job == 'train_data'):
        #    print('Label count : ', label_count,' added! ',k,' times!!')
    #print('Label count : ', label_count,' incremented!')

