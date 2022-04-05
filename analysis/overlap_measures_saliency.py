'''
use ITK to compute Dice overlap between each combination of
[aggregate, white male, black male, white female, black female] average saliency map masks

'''
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import os



def makeParser():
    parser = argparse.ArgumentParser(description='Selecting which dataset to visualize')
    parser.add_argument('--fold', type=str, help='model fold that was used to generate saliency maps')
    parser.add_argument('--exp', type=str, help='experiment name (name of folder with other experiment data)')
    return parser

def itk_nifti_reader(fname):
    #read nifti images with itk reader
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(fname)
    return reader.Execute()

def compute_overlap(image1, image2):
    '''
    compute Dice coefficient overlap between 2 images
    '''
    image1= sitk.Cast(image1, sitk.sitkInt16)
    image2= sitk.Cast(image2, sitk.sitkInt16)

    labelstats = sitk.LabelOverlapMeasuresImageFilter()
    labelstats.Execute(image1,image2)
    dice = labelstats.GetDiceCoefficient()
    return dice

def get_DC_list(group, group_list):
    '''
    get Dice coefficients for combination of "group" + every label in "group_list"
    '''
    DC_list = []
    #read image corresponding to "group"
    fname1 = SOURCE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + EXP + '_' + FOLD + '_' + group + '_registeredAverage.nii.gz'
    image1 = itk_nifti_reader(fname1)

    #loop through every label in "group_list" and compute overlap with "group"
    for g in group_list:
        #read image
        fname2 = SOURCE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + EXP + '_' + FOLD + '_' + g + '_registeredAverage.nii.gz'
        image2 = itk_nifti_reader(fname2)

        dc = compute_overlap(image1, image2)
        print('{}, {}, Dice = {}'.format(group, g, dc))
        DC_list.append(dc)


    return np.array(DC_list).reshape(1, len(DC_list))


parser = makeParser()
args = parser.parse_args()

FOLD = args.fold
EXP = args.exp
CLEANUP_THRESHOLD_VAL = 0.5
SOURCE_DIR = '/Users/emmastanley/Documents/BME/Research/ABCD_sexclassification_2022/' + EXP + '/saliency/' + FOLD + '/thresh_maps/'


groups = ['aggregate_ALL', 'white_TP', 'white_TN', 'black_TP', 'black_TN']
agg_dc = get_DC_list(groups[0], groups)
wm_dc = get_DC_list(groups[1], groups)
wf_dc = get_DC_list(groups[2], groups)
bm_dc = get_DC_list(groups[3], groups)
bf_dc = get_DC_list(groups[4], groups)


#plot Dice values with a heatmap
dc_matrix = np.concatenate((agg_dc, wm_dc, wf_dc, bm_dc, bf_dc), axis=0)

mask = np.zeros_like(dc_matrix)
mask[np.triu_indices_from(mask, k=1)] = 1.00 #diagonal values always equal 1

dc_heatmap = sns.heatmap(dc_matrix, mask=mask, cmap="Purples", cbar = False, annot=True, annot_kws = {'size': 14}, fmt ='.3f')

labels = ['Aggregate', 'White\nmales', 'White\nfemales', 'Black\nmales', 'Black\nfemales']
plt.xticks([0.5, 1.5,2.5, 3.5, 4.5], labels, va='center', fontsize=11)
plt.yticks([0.5, 1.5,2.5, 3.5, 4.5], labels, va='center', fontsize=11)
plt.yticks(rotation=0)

plt.tick_params(axis='x', which='major', pad = 15, bottom=False)
plt.tick_params(axis='y', which='major', left=False)
# plt.title('Subgroup Saliency Map Dice Coefficients', fontsize=16)
# plt.show()
plt.savefig(SOURCE_DIR + 'diceMatrix_' + EXP + FOLD + '.png', dpi=300, bbox_inches="tight")
