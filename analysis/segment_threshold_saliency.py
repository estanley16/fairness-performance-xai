'''
use VTK to clean up extra noise in saliency maps
+ set remaining salient regions = 1 (create a binary salieny map mask)
'''
import os
import vtk
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def makeParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help='experiment name e.g. exp5b', required=True)
    parser.add_argument('--fold', type=str, help='cv fold that saliency maps were generated from e.g. fold4', required=True)
    return parser


def nifti_reader(fname):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader


def global_threshold_segment(image, thresh_lower, thresh_upper):
    '''
    apply global thresholding to assign value of 1 to segmented regions
    (for use when calculating overlap measures)

    inputs:
    -image : vtkImage type
    -thresh_lower: lower threshold for segmentation
    -thresh_upper: upper threshold for segmentation

    returns: thresholded image of vtkImage type
    '''
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(image)
    threshold.ThresholdBetween(thresh_lower, thresh_upper)
    threshold.ReplaceInOn()
    threshold.SetInValue(1) #set values between lower and upper threshold to 1
    threshold.ReplaceOutOn()
    threshold.SetOutValue(0) #set values out of threshold range to 0
    threshold.Update()
    return threshold.GetOutput()

def threshold_below(image, thresh):
    '''
    threshold out only values below the lower limit,
    keep voxel intensity vals above limit as is

    inputs:
    -image : vtkImage type
    -thresh: threshold for segmentation

    returns: thresholded image of vtkImage type
    '''
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputData(image)
    threshold.ThresholdByLower(thresh)
    threshold.ReplaceInOn()
    threshold.SetInValue(0) #set values less than/equal to lower threshold to zero
    threshold.Update()
    return threshold.GetOutput()


def marching_cubes(image):
    '''
    surface rendering with marching cubes applied to image

    inputs: vtkImage type
    returns: vtkMarchingCubes object
    '''
    iso_value = 1 #create one surface
    surface = vtk.vtkMarchingCubes()
    surface.SetInput(image)
    surface.ComputeNormalsOn()
    surface.SetValue(0, iso_value)
    return surface

def render_surface(surface):
    '''
    displays a 3d surface in a render window
    input:
    - surface: surface rendering algorithm (eg. vtkMarchingCubes, vtkFlyingEdges3D)
    '''
    colors = vtk.vtkNamedColors()
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('DarkGray'))

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    mapper = vtk.vtkPolyMapper()
    mapper.SetInputConnection(surface.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('MediumVioletRed'))

    renderer.AddActor(actor)

    renderWindow.Render()
    renderWindowInteractor.Start()


def write_nifti(image, reader, outpath):
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)

    writer.SetNIFTIHeader(reader.GetNIFTIHeader())
    # writer.SetQFac(reader.GetQFac())
    # writer.SetTimeDimension(reader.GetTimeDimension())
    # writer.SetQFormMatrix(reader.GetQFormMatrix())
    # writer.SetSFormMatrix(reader.GetSFormMatrix())

    writer.SetFileName(outpath)
    writer.Update()
    writer.Write()

    return


def segment_save(reader, cleanup_val, SAVE_DIR):
    '''
    applies global thresholding create a binary mask of salient regions + saves nifti
    input:
    - reader: nifti image reader
    - cleanup_val: bottom percent of voxel intensity values to remove (e.g. 0.1 --> keep top 90% of intensity values)
    -SAVE_DIR: path to save the final saliency map to
    '''
    min,max = reader.GetOutput().GetScalarRange()

    UPPER_THRESHOLD = max
    LOWER_THRESHOLD = cleanup_val * UPPER_THRESHOLD
    print(UPPER_THRESHOLD)
    print(LOWER_THRESHOLD)
    #apply global threshold for segmentation
    thresh_img = global_threshold_segment(reader.GetOutput(), LOWER_THRESHOLD, UPPER_THRESHOLD)

    #render if desired
    # surface = marching_cubes(thresh_img)
    # render_surface(surface)

    #save the thresholded nifti
    write_nifti(thresh_img, reader, SAVE_DIR)


    return

def threshold_cleanup_save(reader, cleanup_val, SAVE_DIR):
    '''
    applies just lower thresholding for cleanup + saves nifti
    input:
    - reader: nifti image reader
    - cleanup_val: bottom percent of voxel intensity values to remove (e.g. 0.1 --> keep top 90% of intensity values)
    -SAVE_DIR: path to save the final saliency map to
    '''
    min,max = reader.GetOutput().GetScalarRange()

    UPPER_THRESHOLD = max
    LOWER_THRESHOLD = cleanup_val * UPPER_THRESHOLD

    #apply just lower thresholding for cleanup
    clean_img = threshold_below(reader.GetOutput(), LOWER_THRESHOLD)
    write_nifti(clean_img, reader, SAVE_DIR)

    return



def main():

    #parse arguments
    parser = makeParser()
    args = parser.parse_args()


    CLEANUP_THRESHOLD_VAL = 0.5 #bottom percent of voxel intensity values to remove (e.g. 0.1 --> keep top 90% of intensity values)

    SOURCE_DIR = '/Users/emmastanley/Documents/BME/Research/ABCD_sexclassification_2022/'+ args.exp + '/saliency/' + args.fold + '/'
    SAVE_DIR = SOURCE_DIR + 'thresh_maps/'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    #import avg saliency maps for M, F /  B, W + aggregate & apply threshold processing

    fname1 = args.exp + '_' + args.fold + '_white_TN_registeredAverage.nii.gz'
    img_reader1 = nifti_reader(SOURCE_DIR + fname1)
    segment_save(img_reader1, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname1)
    threshold_cleanup_save(img_reader1, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'clean_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname1)

    fname2 = args.exp + '_' + args.fold + '_white_TP_registeredAverage.nii.gz'
    img_reader2 = nifti_reader(SOURCE_DIR + fname2)
    segment_save(img_reader2, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname2)
    threshold_cleanup_save(img_reader2, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'clean_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname2)

    fname3 = args.exp + '_' + args.fold + '_black_TN_registeredAverage.nii.gz'
    img_reader3 = nifti_reader(SOURCE_DIR + fname3)
    segment_save(img_reader3, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname3)
    threshold_cleanup_save(img_reader3, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'clean_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname3)

    fname4 = args.exp + '_' + args.fold + '_black_TP_registeredAverage.nii.gz'
    img_reader4 = nifti_reader(SOURCE_DIR + fname4)
    segment_save(img_reader4, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname4)
    threshold_cleanup_save(img_reader4, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'clean_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname4)

    fname5 = args.exp + '_' + args.fold + '_aggregate_ALL_registeredAverage.nii.gz'
    img_reader5 = nifti_reader(SOURCE_DIR + fname5)
    segment_save(img_reader5, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'thresh_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname5)
    threshold_cleanup_save(img_reader5, CLEANUP_THRESHOLD_VAL, SAVE_DIR + 'clean_' + str(CLEANUP_THRESHOLD_VAL) + '_' + fname5)



if __name__ == '__main__':
    main()
