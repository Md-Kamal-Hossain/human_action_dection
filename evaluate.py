# -*- coding: utf-8 -*-
import os
from model import *
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import numpy as np
import cv2
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import random

def preprocessing(data):
    data /= 255.
    data -= 0.5
    data *= 2.
    return data

def generate_features(image_path_ori, image_path_crop,model_ori, model_crop):
    feature_dim = 1024
    seq_length = 16
    train_num = 2800
    test_num = 1200
    x_train = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
    x_train_crop = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
    x_val = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
    x_val_crop = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
    y_train = []
    y_val = []

    train_features_ori = np.zeros((train_num,feature_dim),dtype='float32')
    train_features_crop = np.zeros((train_num,feature_dim),dtype='float32')
    train_features = np.zeros((train_num,2*feature_dim),dtype='float32')
    val_features = np.zeros((test_num,2*feature_dim),dtype='float32')
    val_features_ori = np.zeros((test_num,feature_dim),dtype='float32')
    val_features_crop = np.zeros((test_num,feature_dim),dtype='float32')

    actions = os.listdir(image_path_ori)
    actions.sort(key=str.lower)
    train_num_count = 0
    total_count_train = 0
    total_count_val = 0
    label_count = 0
    for action in actions:
        for job in ['train_data','test_data']:
            samples_ori = os.listdir(image_path_ori+action+'/'+job)
            samples_ori.sort(key=str.lower)

            samples_crop = os.listdir(image_path_crop+action+'/'+job)
            samples_crop.sort(key=str.lower)

            if(len(samples_ori) == len(samples_crop)):
                for idx in range(len(samples_ori)):
                    if job == 'train_data':
                        train_num_count += 1
                        images = os.listdir(image_path_ori+action+'/'+job+'/'+samples_ori[idx])
                        images_crop = os.listdir(image_path_crop+ action + '/' + job+'/'+samples_crop[idx])
                        images.sort(key=str.lower)
                        images_crop.sort(key=str.lower)
                        frames_ori = [i for i in range(len(images) - (seq_length-1))]
                        start_frame_ori = random.sample(frames_ori, 1)
                        frames_crop = [i for i in range(len(images_crop) - (seq_length-1))]
                        start_frame_crop = random.sample(frames_crop, 1)
                        # start_frame = [0,16,32]
                        for idx2 in range(len(start_frame_ori)):
                            for j in range(seq_length):
                                img = cv2.imread(image_path_ori+action+'/'+job+'/'+samples_ori[idx]+'/'+images[start_frame_ori[idx2]+j])
                                img_crop = cv2.imread(image_path_crop+action+'/'+job+'/'+samples_crop[idx]+'/'+images_crop[start_frame_crop[idx2]+j])
                                img = cv2.resize(img,(112,112))
                                img_crop = cv2.resize(img_crop,(112,112))
                                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                img_crop = cv2.cvtColor(img_crop,cv2.COLOR_BGR2RGB)
                                x_train[0,j,:,:,:] = img
                                x_train_crop[0,j,:,:,:] = img_crop
                            y_train.append(label_count)
                            x_train = preprocessing(x_train)
                            x_train_crop = preprocessing(x_train_crop)
                            x_train = np.transpose(x_train,(0,2,3,1,4))
                            x_train_crop = np.transpose(x_train_crop,(0,2,3,1,4))
                            feature_ori = model_ori.predict(x_train)
                            feature_crop = model_crop.predict(x_train_crop)
                            train_features_ori[total_count_train,:] = np.array(feature_ori)
                            train_features_crop[total_count_train,:] = np.array(feature_crop)
                            train_features[total_count_train,0:feature_dim] = np.array(feature_ori)
                            train_features[total_count_train,feature_dim:2*feature_dim] = np.array(feature_crop)
                            total_count_train += 1
                            x_train = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
                            x_train_crop = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')

                    else:
                        train_num_count += 1
                        images = os.listdir(image_path_ori + action + '/' + job+'/'+samples_ori[idx])
                        images_crop = os.listdir(image_path_crop + action + '/'+job+'/' + samples_crop[idx])
                        images.sort(key=str.lower)
                        images_crop.sort(key=str.lower)
                        frames_ori = [i for i in range(len(images) - (seq_length-1))]
                        start_frame_ori = random.sample(frames_ori, 1)
                        frames_crop = [i for i in range(len(images_crop) - (seq_length-1))]
                        start_frame_crop = random.sample(frames_crop, 1)
                        # start_frame = [0]
                        for idx2 in range(len(start_frame_ori)):
                            for j in range(seq_length):
                                img = cv2.imread(image_path_ori + action +'/'+job+'/'+ '/' + samples_ori[idx] + '/' + images[frames_ori[idx2]+j])
                                img_crop = cv2.imread(image_path_crop + action +'/'+job+'/'+ '/' + samples_crop[idx] + '/' + images_crop[frames_crop[idx2]+j])
                                img = cv2.resize(img, (112, 112))
                                img_crop = cv2.resize(img_crop, (112, 112))
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
                                x_val[0, j, :, :, :] = img
                                x_val_crop[0,j,:,:,:] = img_crop
                            y_val.append(label_count)
                            x_val = preprocessing(x_val)
                            x_val_crop = preprocessing(x_val_crop)
                            x_val = np.transpose(x_val, (0, 2, 3, 1, 4))
                            x_val_crop = np.transpose(x_val_crop, (0, 2, 3, 1, 4))
                            feature_ori = model_ori.predict(x_val)
                            feature_crop = model_crop.predict(x_val_crop)
                            val_features_ori[total_count_val,:] = np.array(feature_ori)
                            val_features_crop[total_count_val,:] = np.array(feature_crop)
                            val_features[total_count_val, 0:feature_dim] = np.array(feature_ori)
                            val_features[total_count_val, feature_dim:2*feature_dim] = np.array(feature_crop)
                            total_count_val += 1
                            x_val = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
                            x_val_crop = np.zeros((1, seq_length, 112, 112, 3), dtype='float32')
            else:
                print('Error in data. Returning')
                return -1
            label_count += 1
            train_num_count = 0
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    return train_features,y_train,val_features,y_val,train_features_ori,\
           val_features_ori,train_features_crop,val_features_crop


def plot_confusion_matrix(y_true, y_pred, labels, prefix):
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    action_names = ['A001','A002','A003','A004','A005','A006','A007','A008','A009','A010','A011','A012','A013','A014','A015','A016','A017','A018','A019','A020','A021','A022','A023','A024','A025','A026','A027','A028','A029','A030','A031','A032','A033','A034','A035','A036','A037','A038','A039','A040']
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, action_names, rotation=90)
    plt.yticks(xlocations, action_names)
    plt.ylabel('Index of True Classes')
    plt.xlabel('Index of Predict Classes')
    plt.savefig(prefix+'_confusion_matrix.png', dpi=300)
    plt.close()
    # plt.show()


def svc(traindata, trainlabel, testdata, testlabel, labels, prefix):
    print("Start training SVM...")
    svcClf = SVC(C=10.0, kernel="linear", cache_size=3000)
    svcClf.fit(traindata, trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i] == pred_testlabel[i]]) / float(num)
    print("svm Accuracy:", accuracy)
    plot_confusion_matrix(testlabel,pred_testlabel,labels,prefix)


if __name__ == "__main__":
    original_size_image_path = '/datahdd/student/NTU_RGB_Depth/original/'
    crop_image_path = '/datahdd/student/NTU_RGB_Depth/crop/'
    results_savedPath = 'results/'
    _, model_ori = model_original_size()
    _, model_crop = model_original_size()
    model_ori.load_weights(results_savedPath+'original/original_size.h5',by_name=True)
    model_crop.load_weights(results_savedPath+'crop/crop_size.h5',by_name=True)
    train_features,train_labels,val_features,val_labels,\
    train_features_ori,val_features_ori,\
    train_features_crop,val_features_crop = generate_features(original_size_image_path,
                                                                    crop_image_path,
                                                                    model_ori,
                                                                    model_crop)

    train_features = normalize(train_features, norm='l2')
    val_features = normalize(val_features, norm='l2')
    train_features_ori = normalize(train_features_ori, norm='l2')
    val_features_ori = normalize(val_features_ori, norm='l2')
    train_features_crop = normalize(train_features_crop, norm='l2')
    val_features_crop = normalize(val_features_crop, norm='l2')

    nb_classes = 40
    labels = [i for i in range(nb_classes)]
    print("test with global information..")
    svc(train_features_ori,train_labels,val_features_ori,val_labels,labels,'global')

    print("test with partial information..")
    svc(train_features_crop,train_labels,val_features_crop,val_labels,labels,'crop')

    print("test with global and local information..")
    svc(train_features,train_labels,val_features,val_labels,labels,'merge')
    #labels = {0:'walking',....}
    #print(classification_report(y_test,y_predict,target_names = list(labels.values()))) todo: adjust this code line
