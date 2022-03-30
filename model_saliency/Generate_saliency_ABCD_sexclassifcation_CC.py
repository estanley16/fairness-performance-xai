#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:18:10 2021
Modified on Wed Feb 23 14:34:07 2022

@author: emmastanley
NARVAL CODE
"""

# have to adapt tf-keras-vis commands to 0.5.5 version (that's what currently available on CC)

import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.processing
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import tf_keras_vis
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
import os

# from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
# from tf_keras_vis.utils.scores import BinaryScore

def load_image_data(data_directory, ids_to_load, img_dim):
    '''
    inputs:
        data_directory: directory containing images
        ids_to_load: np array of ids to load
        img_dim: tuple of x,y,z image dimensions
    outputs:
        data: tensor containing image data for each patient
    '''

    (xdim, ydim, zdim) = img_dim
    patient_list = ids_to_load.tolist()
    data = np.zeros((len(patient_list), xdim, ydim, zdim, 1)) #placeholder tensor for image data
    affines = np.zeros((len(patient_list), 4, 4)) #placeholder tensor for affine arrays

    i = 0
    for ID in patient_list:
        filename_full = data_directory + str(ID) + '/' + str(ID) + '_rigidreg_T1strip_Warped.nii.gz'
        nib_img = nib.load(filename_full)
        affines[i,:,:]=nib_img.affine

        img = np.array(nib_img.dataobj)
        scale=np.max(img[:])-np.min(img[:])
        img=(img-np.min(img[:]))/scale
        img = img - 0.5 #all values between [-0.5,0.5]

        data[i,:,:,:,0] = img
        i += 1
    return data, affines

def subsample(array, num):
    """
    function for subsampling ID arrays to a constant value
    allieviate memory requirements and ensure both subgroups and sexes have the same amnt of representation
    convert np array back to pandas series in order to sample easily with a random seed, then back to np array
    """
    series = pd.Series(array)
    series_sample = series.sample(n=num, random_state=6)
    array_sample = series_sample.to_numpy()
    return array_sample



#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--array', type=str, help='array to load (TP, TN, FN or FP')
parser.add_argument('--fold', type=str, help='fold from experiment to generate saliency on')
parser.add_argument('--exp', type=str, help='experiment name')
parser.add_argument('--subgroup', type=str, help='white, black, or aggregate')
parser.add_argument('--modelname', type=str, help='name of model to generate saliency maps for')
args = parser.parse_args()


#parameters and directory paths
N_MAPS = 20 #number of subjects of each TP/TN/FP/FN array to generate saliency maps for
SMOOTH_SAMPLES = 20 #number of smoothgrad iterations
SMOOTH_NOISE = 0.2 #smoothgrad gaussian noise spread

abcd_directory = '/home/estanley/scratch/abcd_sexclassification' #main directory
img_directory = '/home/estanley/scratch/abcd_miccai/rigid_processed_originalAtlas/' #directory with images
saliency_dir = abcd_directory + '/saliency_maps/'
saliency_subdir = saliency_dir + args.exp + '_' + args.fold + '_' + args.subgroup + '_' + args.array + '/' #eg: ./exp5_fold1_white_TP/
if not os.path.exists(saliency_subdir):
    os.makedirs(saliency_subdir)


#retrieve dataframe with predictions for desired fold from data folder
df_path = abcd_directory + '/data/' + args.exp + '_' + args.fold + '_df.csv'
df = pd.read_csv(df_path, index_col=0)
col_name = 'preds_' + args.fold
df = df[~df[col_name].isnull()] #get rid of rows without values for that fold
df[col_name] = df[col_name].str[1].astype(int) #get rid of brackets + convert to numeric


#slice dataframe depending on which subgroup we are generating saliency maps for
if args.subgroup == 'white':
    df = df.loc[df['white_only'] == 1]
elif args.subgroup == 'black':
    df = df.loc[df['black_only'] == 1]
elif args.subgroup == 'aggregate':
    df = df
    N_MAPS = 10 #run the script separately for TP and TN for the aggregate map, will generate 10xTP 10xTN

#convert true classes and preds to np arrays and get indices of TP/TN/FP/FN
#male = 1 = "positive", female = 0 = 'negative'
y_true = df['M'].to_numpy()
y_pred = df[col_name].to_numpy()

TP_idx = np.where((y_pred ==1) & (y_true==1))
TN_idx = np.where((y_pred ==0) & (y_true==0))
FP_idx = np.where((y_pred == 1) & (y_true == 0))
FN_idx = np.where((y_pred == 0) & (y_true == 1))

test_ids = df.iloc[:,0].to_numpy() #array of all subject IDs

#arrays of subject IDs by classification
TP_ids_full = test_ids[TP_idx]
TN_ids_full = test_ids[TN_idx]
FP_ids_full = test_ids[FP_idx]
FN_ids_full = test_ids[FN_idx]

TP_ids = subsample(TP_ids_full, N_MAPS)
TN_ids = subsample(TN_ids_full, N_MAPS)
# FP_ids = subsample(FP_ids_full, N_MAPS) #commented out bc black subjects dont have enough N
# FN_ids = subsample(FN_ids_full, N_MAPS)

#load model to generate saliency maps for
model_dir = model_dir = abcd_directory + '/weights/' + args.modelname
model = tf.keras.models.load_model(model_dir)
input_dims = (197, 233, 189)
model.summary()

#load image data and score functions corresponding to the tf-keras-vis docs
if args.array=='TP':
    array, affines = load_image_data(img_directory, TP_ids, input_dims)
    array_IDs = TP_ids

    def score_function(output): # output shape is (batch_size, 1)
        return output[:, 0] #for positive case

elif args.array=='FP':
    array, affines = load_image_data(img_directory, FP_ids, input_dims)
    array_IDs = FP_ids

    #generates map for positive class, which it thought it was
    def score_function(output): # output shape is (batch_size, 1)
        return output[:, 0] #for positive case

elif args.array=='TN':
    array, affines = load_image_data(img_directory, TN_ids, input_dims)
    array_IDs = TN_ids

    def score_function(output): # output shape is (batch_size, 1)
        return -1.0 * output[:, 0] #for negative case

elif args.array=='FN':
    array, affines = load_image_data(img_directory, FN_ids, input_dims)
    array_IDs = FN_ids

    #generates map for negative class, which it thought it was
    def score_function(output): # output shape is (batch_size, 1)
        return -1.0 * output[:, 0] #for negative case


print('{} array length: {}'.format(args.array, len(array)))

def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=model_modifier_function,
                    clone=True)


for i, ID in enumerate(array_IDs):
    print('generating saliency map for {}'.format(ID))
    saliency_map = saliency(score_function,
                        array[i,:,:,:,:],
                        smooth_samples=SMOOTH_SAMPLES, # The number of calculating gradients iterations.
                        smooth_noise=SMOOTH_NOISE) # noise spread level.

    # Since v0.6.0, calling `normalize()` is NOT necessary.
    saliency_map = normalize(saliency_map)
    smoothgrad_img = nib.Nifti1Image(saliency_map[0,:,:,:]*255, affine=affines[i,:,:])
    nib.save(smoothgrad_img, saliency_subdir + ID + '.nii.gz')
