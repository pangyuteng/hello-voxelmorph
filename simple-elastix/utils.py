raise NotImplementedError()


import os
import sys
import logging
logger = logging.getLogger('utils')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

import traceback
import imageio
import pydicom 
import SimpleITK as sitk
import numpy as np
import numpy.ma as ma
from scipy import ndimage
from skimage import measure

def imread(mypath):
    reader= sitk.ImageFileReader()
    reader.SetFileName(mypath)
    img_obj = reader.Execute()
    return img_obj

# https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# naive lungseg using image processing methods.
def lung_seg(img_obj):
    
    arr = sitk.GetArrayFromImage(img_obj)
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()

    bkgd = np.zeros(arr.shape).astype(np.uint8)
    pad = 5
    bkgd[:,:,:pad]=1
    bkgd[:,:,-1*pad:]=1
    bkgd[:,:pad,:]=1
    bkgd[:,-1*pad:,:]=1
    
    # assume < -300 HU are voxels within lung
    procarr = (arr < -300).astype(np.int)
    procarr = ndimage.morphology.binary_closing(procarr,iterations=1)

    label_image, num = ndimage.label(procarr)
    region = measure.regionprops(label_image)

    region = sorted(region,key=lambda x:x.area,reverse=True)
    lung_mask = np.zeros(arr.shape).astype(np.uint8)
    
    # assume `x` largest air pockets except covering bkgd is lung, increase x for lung with fibrosis (?)
    x=2
    for r in region[:x]: # should just be 1 or 2, but getting x, since closing may not work.
        mask = label_image==r.label
        contain_bkgd = np.sum(mask*bkgd) > 0
        if contain_bkgd > 0:
            continue
        lung_mask[mask==1]=1

    lung_mask = ndimage.morphology.binary_closing(lung_mask,iterations=5)

    lung_obj = sitk.GetImageFromArray(lung_mask.astype(arr.dtype))
    lung_obj.SetSpacing(spacing)
    lung_obj.SetOrigin(origin)
    lung_obj.SetDirection(direction)

    return lung_obj
    
def register(tlc_path,rv_path,work_dir,save_deformation=False):

    logger.debug('imread...')
    fixed_path = rv_path
    moving_path = tlc_path

    fixedImage = imread(fixed_path)
    movingImage = imread(moving_path)
    logger.debug(f'{movingImage.GetSize()},{fixedImage.GetSize()}')
    logger.debug(f'{movingImage.GetOrigin()},{fixedImage.GetOrigin()}')
    logger.debug(f'{movingImage.GetDirection()},{fixedImage.GetDirection()}')
        
    logger.debug('init...')

    logger.debug('read params...')
    parameter_map_0 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/affine.txt"))
    parameter_map_1 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/bspline1.txt"))
    parameter_map_2 = sitk.ReadParameterFile(os.path.join(THIS_DIR,"param/bspline2.txt"))

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(parameter_map_0)
    parameterMapVector.append(parameter_map_1)
    parameterMapVector.append(parameter_map_2)

    logger.debug(f"registering... ")
    elastixImageFilter = sitk.ElastixImageFilter()
    logdir = os.path.join(work_dir,"elastix")
        
    elastixImageFilter.LogToFileOn()        
    os.makedirs(logdir,exist_ok=True)
    elastixImageFilter.SetOutputDirectory(logdir)

    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)

    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    registered_obj = elastixImageFilter.GetResultImage()

    if save_deformation:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(movingImage)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.LogToConsoleOn()
        transformixImageFilter.SetOutputDirectory(logdir)
        transformixImageFilter.Execute()
        deformation_file = os.path.join(work_dir,'deformation.nii.gz')
        sitk.WriteImage(transformixImageFilter.GetDeformationField(),deformation_file)

    return registered_obj


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def plot_prm(
        mydict,lung_mask,tlc,rv,
        tlc_val_mask,rv_val_mask,
        normal_lung,pd,fsad,emphysema,
        spacing,img_path,grid_on=False):

    valid = np.logical_and(tlc_val_mask,rv_val_mask)
    H, xedges, yedges = np.histogram2d(rv[valid], tlc[valid], bins=(100,100),range=[[-1000,-500],[-1000,-500]])
    H = H.T

    fig = plt.figure(figsize=(15,5))
    
    ax = fig.add_subplot(131, title=f'TLC, RV joint histogram')
    plt.imshow(H, interpolation='nearest', origin='lower', cmap='Greys',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # emphysema
    p = Rectangle( (-1000,-1000),144,50)
    pc = PatchCollection([p],facecolor='red',alpha=0.2)
    ax.add_collection(pc)
    val = int(mydict["emphysema"]*100)
    ax.text(-995, -990, f'emphysema {val}%')

    # fsad
    p = Rectangle( (-1000,-950),144,140)
    pc = PatchCollection([p],facecolor='yellow',alpha=0.2)
    ax.add_collection(pc)
    val = int(mydict["fuctional_small_airway_disease"]*100)
    ax.text(-995, -830, f'fSAD {val}%')

    # normal
    p = Rectangle( (-856,-950),356,140)
    pc = PatchCollection([p],facecolor='green',alpha=0.2)
    ax.add_collection(pc)    
    val = int(mydict["normal"]*100)
    ax.text(-510, -830, f'normal {val}%',horizontalalignment='right')

    # parenchymal disease
    p = Rectangle( (-1000,-810),500,350)
    pc = PatchCollection([p],facecolor='purple',alpha=0.2)
    ax.add_collection(pc)
    val = int(mydict["parenchymal_disease"]*100)
    ax.text(-510, -800, f'parenchymal disease {val}%',horizontalalignment='right')

    # na
    val = int(mydict["na"]*100)
    ax.text(-510, -990, f'na {val}%',horizontalalignment='right')

    plt.xlabel('Expiration (HU)')
    plt.ylabel('Inspiration (HU)')
    
    # get mid coronal slice
    z,x,y = np.where(lung_mask==1)
    slice_x = int(np.median(x))    
    aspect = spacing[2]/spacing[0]
    tlc_slice = tlc[:,slice_x,:].squeeze()
    rv_slice = rv[:,slice_x,:].squeeze()

    alpha = 0.7
    normal_slice = normal_lung[:,slice_x,:].squeeze().astype(np.uint8)
    cmap_normal = matplotlib.colors.ListedColormap([(1,1,1,0),(0,1,0,alpha)],name='normal')

    pd_slice = pd[:,slice_x,:].squeeze().astype(np.uint8)
    cmap_pd = matplotlib.colors.ListedColormap([(1,1,1,0),(.5,0,.5,alpha)],name='pd')

    emph_slice = emphysema[:,slice_x,:].squeeze().astype(np.uint8)
    cmap_emph = matplotlib.colors.ListedColormap([(1,1,1,0),(1,0,0,alpha)],name='emph')

    fsad_slice = fsad[:,slice_x,:].squeeze().astype(np.uint8)
    cmap_fsad = matplotlib.colors.ListedColormap([(1,1,1,0),(1,1,0,alpha)],name='fsad')

    ax2 = fig.add_subplot(232, title=f'TLC (registered)')
    ax2.imshow(tlc_slice,vmin=-600-(1500/2),vmax=-600+(1500/2),cmap='gray',interpolation='none',aspect=aspect)
    if grid_on:
        ax2.grid(color='b', linestyle='-', linewidth=0.5)
    else:
        ax2.axis('off')

    ax2 = fig.add_subplot(235, title=f'TLC (registered)')
    ax2.imshow(tlc_slice,vmin=-600-(1500/2),vmax=-600+(1500/2),cmap='gray',interpolation='none',aspect=aspect)
    ax2.imshow(normal_slice,cmap=cmap_normal, interpolation='none', aspect=aspect)
    ax2.imshow(pd_slice,cmap=cmap_pd, interpolation='none', aspect=aspect)
    ax2.imshow(emph_slice,cmap=cmap_emph, interpolation='none', aspect=aspect)
    ax2.imshow(fsad_slice,cmap=cmap_fsad, interpolation='none', aspect=aspect)
    
    if grid_on:
        ax2.grid(color='b', linestyle='-', linewidth=0.5)
    else:
        ax2.axis('off')

    ax3 = fig.add_subplot(233, title=f'RV')
    ax3.imshow(rv_slice,vmin=-600-(1500/2),vmax=-600+(1500/2),cmap='gray',interpolation='none',aspect=aspect)
    if grid_on:
        ax3.grid(color='b', linestyle='-', linewidth=0.5)
    else:
        ax3.axis('off')

    ax3 = fig.add_subplot(236, title=f'RV')
    ax3.imshow(rv_slice,vmin=-600-(1500/2),vmax=-600+(1500/2),cmap='gray',interpolation='none',aspect=aspect)
    ax3.imshow(normal_slice,cmap=cmap_normal, interpolation='none', aspect=aspect)
    ax3.imshow(pd_slice,cmap=cmap_pd, interpolation='none', aspect=aspect)
    ax3.imshow(emph_slice,cmap=cmap_emph, interpolation='none', aspect=aspect)
    ax3.imshow(fsad_slice,cmap=cmap_fsad, interpolation='none', aspect=aspect)
    
    if grid_on:
        ax3.grid(color='b', linestyle='-', linewidth=0.5)
    else:
        ax3.axis('off')    

    plt.savefig(img_path)    
    plt.close()
