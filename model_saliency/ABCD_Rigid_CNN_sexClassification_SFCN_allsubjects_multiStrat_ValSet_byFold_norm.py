#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:14:09 2021

@author: emmastanley

NARVAL CODE
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.processing
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report,  accuracy_score
from tensorflow.keras import backend as K

#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='number of traning epochs to run')
parser.add_argument('--output', type=str, help='output prefix to store weights and log files')
parser.add_argument('--fold_no', type=int, help='which cross val fold to run')


args = parser.parse_args()


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size, dim, shuffle, imageDirectory, covariates, target_col_position):
        'Initialization'
        #image dimension
        self.dim = dim
        self.batch_size = batch_size
        #list_IDs is a numpy array that contains the patient ids.
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        # Here the images are stored under imageDirectory/<id>/<id>_T1strip.nii.gz
        self.imageDirectory = imageDirectory
        # covariates is a dataframe that contains the patient ids and associated outcomes
        self.covariates = covariates
        self.on_epoch_end()
        # target col position is the column index of the outcome of interest
        self.target_col_position = target_col_position



    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
       # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_im = np.empty((self.batch_size, *self.dim, 1))# contains images
        y = np.empty((self.batch_size), dtype="int") # contains the output

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
             nib_img = nib.load(self.imageDirectory + str(ID) + '/' + str(ID) + '_rigidreg_T1strip_Warped.nii.gz')
             img = np.array(nib_img.dataobj)
             scale=np.max(img[:])-np.min(img[:])
             img=(img-np.min(img[:]))/scale #all values between [0,1]
             img = img - 0.5 #all values between [-0.5,0.5]
             X_im[i,] = np.float32(img.reshape(self.dim[0],self.dim[1],self.dim[2],1))

            # Store class
             y[i] = self.covariates.iloc[np.where(self.covariates.iloc[:,0]==ID)].iloc[0,self.target_col_position] #where row = ID, select the corresponding score from col of interest
             y[i] = np.float32(y[i])

        return X_im,y



def sfcn(inputLayer):
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="conv1")(inputLayer)
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)

    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)

    #block 7, different from paper
    x=AveragePooling3D()(x)
    x=Dropout(.2)(x)
    x=Flatten(name="flat1")(x)
    x=Dense(units=1, activation='sigmoid',name="dense1")(x)

    return x



def ConfusionMatrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp = disp.plot(include_values=True,cmap='Blues', ax=None, xticks_rotation='horizontal')
    plt.title('Confusion Matrix')
    cmname = abcd_directory + '/plots/' + args.output + '_cm.png'
    plt.savefig(cmname)


##########################################################################################
##########################################################################################

abcd_directory = '/home/estanley/scratch/abcd_sexclassification'
img_directory = '/home/estanley/scratch/abcd_miccai/rigid_processed_originalAtlas/'

input_dims = (197, 233, 189)
BATCH_SIZE=2
exp_name = 'train_val_test' #experiment name to get the splits from

main_df = pd.read_csv(abcd_directory + '/data/t1_img_demographics.csv', index_col=0)
target_col_position = main_df.columns.get_loc('M')



#load train, val, and test indices for this split
train = pickle.load(open(abcd_directory + '/splits/'+ exp_name + '/train_'+ str(args.fold_no), 'rb'))
val = pickle.load(open(abcd_directory + '/splits/'+ exp_name + '/val_'+ str(args.fold_no), 'rb'))
test = pickle.load(open(abcd_directory + '/splits/'+ exp_name + '/test_' + str(args.fold_no), 'rb'))


#generate the train and test data for this fold by selecting subsets of main dataframe
train_df = main_df.loc[train]
val_df = main_df.loc[val]
test_df = main_df.loc[test]

train_generator = DataGenerator(train_df.iloc[:,0].to_numpy(), BATCH_SIZE, input_dims, True, img_directory, train_df, target_col_position)
val_generator = DataGenerator(val_df.iloc[:,0].to_numpy(), BATCH_SIZE, input_dims, True, img_directory, val_df, target_col_position)
test_generator = DataGenerator(test_df.iloc[:,0].to_numpy(), BATCH_SIZE, input_dims, False, img_directory, test_df, target_col_position)


#create model
inputA = Input(shape=(input_dims[0], input_dims[1], input_dims[2], 1), name="Input")
z = sfcn(inputA)
model = Model([inputA], z)
model.summary()
#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.003), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['acc', Precision(), Recall(), AUC()])


fname = abcd_directory + '/weights/' + args.output + '_fold' + str(args.fold_no) + '_{epoch:02d}-{val_loss:.2f}'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(fname, monitor='val_loss', save_best_only=True)


print('------------------------------------------------------------------------')
print(f'Training for fold {args.fold_no} ...')

history = model.fit(train_generator,
                    epochs = args.epochs,
                    validation_data=val_generator,
                    callbacks = [checkpoint_cb],
                    verbose=2)

#plot loss
pname = abcd_directory + '/plots/' + args.output + '_fold' + str(args.fold_no) + '_loss.png'
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(pname)
plt.clf()

#plot acc
pname = abcd_directory + '/plots/' + args.output + '_fold' + str(args.fold_no) + '_acc.png'
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(pname)
plt.clf()



#########################
#evaluate classifier
print('------------------------------------------------------------------------')
print(f'Evaluating classifier for fold {args.fold_no} ...')

y_pred_raw = model.predict(test_generator)
print('Model output: {}'.format(y_pred_raw))

#convert sigmoid output to classes
threshold = 0.5
y_pred_bool = (y_pred_raw>=0.5)
y_pred = y_pred_bool.astype(int)
# print('Predictions: {}'.format(y_pred))

#get true labels
y_true_all= test_df.loc[:,'M']
y_true = y_true_all.iloc[:len(y_pred)] #match the ground truth array to the size of the test array
y_true_np = y_true.to_numpy().reshape(-1,1) #get in numpy array form for computing metrics with y_pred


labels = ['female', 'male']

print("Classification Report: \n",
      classification_report(y_true_np, y_pred, target_names=labels))

acc = accuracy_score(y_true_np, y_pred)
print('Accuracy: {}'.format(acc))

#dataframe of true and predicted values

df = y_true.to_frame() #the pandas series version of y_true
df.rename(columns={'M': 'true'}, inplace=True)
df['pred']=y_pred.tolist()

#add predictions to this fold to dataframe with demographics
col_name = 'preds_fold' + str(args.fold_no)
main_df[col_name]=df[['pred']]

#save full model
model.save(abcd_directory + '/weights/' + args.output + '_fold' + str(args.fold_no) + '_model')

#save dataframe with fold predictions
main_df.to_csv(abcd_directory + '/data/' + args.output + '_df.csv')

ConfusionMatrix(y_true, y_pred)
