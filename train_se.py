from model import *
# from augmentor import *
# Sys
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model
# Backenda
from keras import backend as K
# Utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.utils import multi_gpu_model
# Trains
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau,CSVLogger

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

 # etc
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS_PATH = '/home/wonjae/YongHyeok/Classification_TASK3/weights/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'
# WEIGHTS_PATH_NO_TOP = 'https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_no_top.h5'
IMAGE_SIZE = 299

# WEIGHTS_PATH = '/home/wonjae/Classification_inception_v4/weights/Inception4_classsn_lr_05_1o2_50ep.hdf5'

weights_path = get_file('inception-v4_weights_tf_dim_ordering_tf_kernels.h5',WEIGHTS_PATH,cache_subdir='models')

# weights_path = get_file('Inception4_classsn_lr_05_1o2_50ep.hdf5',WEIGHTS_PATH,cache_subdir='models')

model = inception_v4(num_classes=1001, dropout_keep_prob=0.2, weights=None, include_top=True)

 # Load pretrained model weights by imageNet
model.load_weights(weights_path, by_name=True)

 # Adjust the output node
model.layers.pop()
model.outputs = model.layers[-1].output
model.layers[-1].outbound_nodes = []
x = Dense(7, activation='softmax')(model.outputs)
model = Model(inputs=model.input, outputs=x)



 # Set a mult-gpu model
model = multi_gpu_model(model, gpus=3)


# model.load_weights('/home/wonjae/Classification_inception_v4/weights/Inception4_classsn_lr_06_2o2_60ep.hdf5', by_name=True)


'''
one half of data
''' 
 # Load data
images_1 = np.load('Pre-Processing/train_images_task3_299_aug_ran_1o2.npy')
gt_labels_1 = np.load('Pre-Processing/gt_task3_299_aug_ran_1o2.npy')

data_size = images_1.shape[0]
 # Divide dataset for training and validation// this is indexing numbers
train_indices_1 = np.random.choice(data_size,data_size/100*95,replace=False)
print("Number of Trainning data set is :  ",len(train_indices_1))
val_indices_1 = [i for i in range(data_size) if i not in train_indices_1]
print("Number of Validation data set is : ",len(val_indices_1))

# training set
train_images_1=images_1[train_indices_1,:,:]
train_labels_1=gt_labels_1[train_indices_1]
del train_indices_1
# validation set
val_images_1 = images_1[val_indices_1,:,:]
val_labels_1 = gt_labels_1[val_indices_1]
del images_1
del gt_labels_1
del val_indices_1

print ("Start calculation of data normalization in one half")
train_mean_1 = np.mean(train_images_1,axis = (0,1,2,3))
train_std_1 = np.std(train_images_1,axis = (0,1,2,3))
train_images_norm_1 = (train_images_1 - train_mean_1)/(train_std_1+1e-7)
del train_images_1

val_mean_1 = np.mean(val_images_1,axis=(0,1,2,3))
val_std_1 = np.std(val_images_1,axis = (0,1,2,3))
val_images_norm_1 = (val_images_1-val_mean_1)/(val_std_1+1e-7)
del val_images_1

print ("End calculation of data normalization")

# Training!
model.compile(optimizer = rmsprop(lr=0.65, rho=0.9, epsilon=1.0, decay=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)

csv_logger_1 = CSVLogger('Inception4_classsn_lr_065_1o2_50ep.csv') # Training log

model_checkpoint_1 = ModelCheckpoint("Inception4_classsn_lr_065_1o2_50ep.hdf5",monitor = 'val_loss',verbose = 1,save_best_only=True)

model.fit(train_images_norm_1, train_labels_1, batch_size=64, epochs=60, verbose=1,validation_data=(val_images_norm_1,val_labels_1), shuffle=True, callbacks=[csv_logger_1,model_checkpoint_1])
# model.fit(train_images_1, train_labels_1, batch_size=64, epochs=1, verbose=1,validation_data=(val_images_1,val_labels_1), shuffle=True, callbacks=[csv_logger_1,model_checkpoint_1])
del train_images_norm_1
del train_labels_1
del val_images_norm_1
del val_labels_1

del model_checkpoint_1
        
'''
Last half of data
''' 
 # Load data
images_2 = np.load('Pre-Processing/train_images_task3_299_aug_ran_2o2.npy')
gt_labels_2 = np.load('Pre-Processing/gt_task3_299_aug_ran_2o2.npy')

data_size = images_2.shape[0]
model.load_weights('/home/wonjae/YongHyeok/Classification_TASK3/Inception4_classsn_lr_065_1o2_50ep.hdf5', by_name=True)

 # Divide dataset for training and validation// this is indexing numbers
train_indices_2 = np.random.choice(data_size,data_size/100*95,replace=False)
print("Number of Trainning data set is :  ",len(train_indices_2))
val_indices_2 = [i for i in range(data_size) if i not in train_indices_2]
print("Number of Validation data set is : ",len(val_indices_2))

# training set
train_images_2=images_2[train_indices_2,:,:]
train_labels_2=gt_labels_2[train_indices_2]
del train_indices_2
# validation set
val_images_2 = images_2[val_indices_2,:,:]
val_labels_2 = gt_labels_2[val_indices_2]
del images_2
del gt_labels_2
del val_indices_2

print ("Start calculation of data normalization in Last half")
train_mean_2 = np.mean(train_images_2,axis = (0,1,2,3))
train_std_2 = np.std(train_images_2,axis = (0,1,2,3))
train_images_norm_2 = (train_images_2 - train_mean_2)/(train_std_2+1e-7)
del train_images_2

val_mean_2 = np.mean(val_images_2,axis=(0,1,2,3))
val_std_2 = np.std(val_images_2,axis = (0,1,2,3))
val_images_norm_2 = (val_images_2-val_mean_2)/(val_std_2+1e-7)
del val_images_2

print ("End calculation of data normalization")

# Training!
model.compile(optimizer = rmsprop(lr=0.65, rho=0.9, epsilon=1.0, decay=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])
lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=5, min_lr=0.5e-6)

csv_logger_2 = CSVLogger('Inception4_classsn_lr_065_2o2_50ep.csv') # Training log

model_checkpoint_2 = ModelCheckpoint("Inception4_classsn_lr_065_2o2_50ep.hdf5",monitor = 'val_loss',verbose = 1,save_best_only=True)

model.fit(train_images_norm_2, train_labels_2, batch_size=64, epochs=60, verbose=1,validation_data=(val_images_norm_2,val_labels_2), shuffle=True, callbacks=[csv_logger_2,model_checkpoint_2])
# model.fit(train_images_2, train_labels_2, batch_size=64, epochs=1, verbose=1,validation_data=(val_images_2,val_labels_2), shuffle=True, callbacks=[csv_logger_2,model_checkpoint_2])
del train_images_norm_2
del train_labels_2
del val_images_norm_2
del val_labels_2