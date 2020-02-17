import os
import numpy as np
import cv2
import random

def preprocessing(data):
    data /= 255.
    data -= 0.5
    data *= 2.
    return data


# original size 112x112, crop size 64x64
def load_data(image_path, flag):
    if flag == 'original':
        size = 112
    elif flag == 'crop':
        size = 112
    else:
        raise ('only support original and crop!')

    # There are 40 actions or labels
    # for training - 70 videos inside each action
    # for testing - 30 videos inside each action

    # we take 16 consecutive images to define a sequence
    seq_length = 16
    # for training - 40*70
    x_train = np.zeros((2800,seq_length,size,size,3),dtype='float32')
    y_train = []
    #for testing - 40*30
    x_val = np.zeros((1200,seq_length,size,size,3),dtype='float32')
    y_val = []

    actions = os.listdir(image_path)
    actions.sort(key=str.lower)
    train_num_count = 0
    total_count_train = 0
    total_count_val = 0
    label_count = 0
    for action in actions:
        for job in ['train_data','test_data']:
            samples = os.listdir(image_path+action+'/'+job)
            samples.sort(key=str.lower)#
            print('No. of samples in '+action+' in '+job+' : ', len(samples))
            #k = 0
            for sample in samples:
                if job == 'train_data':
                    train_num_count += 1
                    images = os.listdir(image_path+action+'/'+job+'/'+sample)
                    images.sort(key=str.lower)
                    #print('No. of images in '+action+' in '+job+' in '+sample+' : ', len(images))
                    frames = [i for i in range(len(images)-(seq_length-1))]
                    #print('No. of frames in '+action+' in '+job+' in '+sample+' : ', len(frames))
                    start_frame = random.sample(frames,1)
                    #print('No. of start frames in '+action+' in '+job+' in '+sample+' : ', len(start_frame))
                    for frame in start_frame:
                        for j in range(seq_length):
                            img = cv2.imread(image_path+action+'/'+job+'/'+sample+'/'+images[frame+j])
                            img = cv2.resize(img,(size,size))
                            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                            x_train[total_count_train,j,:,:,:] = img
                            y_train.append(label_count)
                            total_count_train += 1
                            
                            #k += 1
                    
                            
                else:
                    train_num_count += 1
                    images = os.listdir(image_path+action+'/'+job+'/'+sample)
                    images.sort(key=str.lower)
                    #print('No. of images in '+action+' in '+job+' in '+sample+' : ', len(images))
                    frames = [i for i in range(len(images) - (seq_length-1))]
                    #print('No. of frames in '+action+' in '+job+' in '+sample+' : ', len(frames))
                    start_frame = random.sample(frames, 1)
                    for frame in start_frame:
                        for j in range(seq_length):
                            img = cv2.imread(image_path+action+'/'+job+'/'+sample+'/'+images[frame+j])
                            img = cv2.resize(img, (size, size))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            x_val[total_count_val, j, :, :, :] = img
                            y_val.append(label_count)
                            total_count_val += 1
            #if(job == 'train_data'):
            #    print('Label count : ', label_count,' added! ',k,' times!!')
        label_count += 1
        #print('Label count : ', label_count,' incremented!')

        train_num_count = 0
    x_train = preprocessing(x_train)
    x_train = np.transpose(x_train,(0,2,3,1,4))
    x_val = preprocessing(x_val)
    x_val = np.transpose(x_val,(0,2,3,1,4))
    #print(len(x_train), len(y_train), len(x_val), len(y_val))

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    #print(len(x_train), len(y_train), len(x_val), len(y_val))

    return x_train,y_train,x_val,y_val
