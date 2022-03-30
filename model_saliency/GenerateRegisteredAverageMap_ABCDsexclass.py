import numpy as np
import pandas as pd
import SimpleITK as sitk
import subprocess
import os
import argparse
#NARVAL code


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--array', type=str, help='array to load (TP, TN, FN or FP), ALL if aggregate')
parser.add_argument('--fold', type=str, help='fold from experiment to generate saliency on')
parser.add_argument('--exp', type=str, help='experiment name')
parser.add_argument('--subgroup', type=str, help='white, black, or aggregate')
args = parser.parse_args()

abcd_directory = '/home/estanley/scratch/abcd_sexclassification' #main directory
image_dir = '/home/estanley/scratch/abcd_miccai/rigid_processed_originalAtlas/' #directory with rigid registered images, contains rigid transform matrices
saliency_dir = abcd_directory + '/saliency_maps/'
maps_dir = saliency_dir + args.exp + '_' + args.fold + '_' + args.subgroup + '_' + args.array + '/' #eg: ./exp5_fold1_white_TP/
atlas_dir = '/home/estanley/scratch/abcd_miccai/atlas/nihpd_asym_07.5-13.5_t1w_stripped.nii.gz'

#for aggregate -> move all images into "ALL" folder

array_list = [args.array]

for label in array_list:

    count=0
    for img in os.listdir(maps_dir):
        ID = img[:15]
        print('ID')
        # apply tranform the non linear transform to transform the map into the NIHPD atlas space
        #(input images were already affinely registered to the MNI but if it is not the case for you, then apply GenericeAffine.mat + Warp.nii.gz)
        subprocess.check_output(['antsApplyTransforms -d 3 -e 0 -i '+maps_dir+'/'+img+ ' -o '+maps_dir+'MapTransformed.nii.gz -t '+image_dir+ID+'/'+ID+'_deformreg_T1strip_0GenericAffine.mat -t '+image_dir+ID+'/'+ID+'_deformreg_T1strip_1Warp.nii.gz -n Linear -r '+atlas_dir],shell=True)
        # read transformed map
        t1 = sitk.ReadImage(maps_dir+'MapTransformed.nii.gz')
        img_t1=sitk.GetArrayFromImage(t1)
        # add to sum map
        if(count==0):
            summap=img_t1
        else:
            summap=img_t1+summap

        count=count+1
        print('array:{}, read image:{}, ID: {}'.format(label,count,ID))

    # divide by number of maps to average
    summap=summap/count
    average=sitk.GetImageFromArray(summap)
    #Copy info to keep the orientation info
    average.CopyInformation(t1)
    sitk.WriteImage(average, maps_dir + args.exp + '_' + args.fold + '_' + args.subgroup + '_' + args.array +'_registeredAverage.nii.gz')
