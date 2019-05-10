#Imports
import PIL
import numpy as np
import tifffile as tf
from scipy.ndimage import gaussian_filter
import skimage as ski
import time
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
import matplotlib.pyplot as plt
import csv

#Constants
voxel = [0.225,0.225,2] #x,y,z in um

gaussianblur = [1, 1, 1] #px (after downsampling)
threshold = 600 #grey values
bbox_zstack = False # show all squares (True) or wire frame (False)
pad = [30, 30, 5] #px, padding in x, y, z, for bounding box 

#Import_Volume
root = tk.Tk()
root.withdraw()
files_path = askdirectory()
save_path = files_path + "/processed/"
os.mkdir(save_path)
os.mkdir(save_path+"rois/")

start = time.time()
files = next(os.walk(files_path))[2] #all files within directory
files.sort()

im = tf.imread(files_path + "/" + files[5])
im = np.zeros([im.shape[0], im.shape[1], len(files)]) #make empty image
               
i = 0

for file in files:
    if (file[-4:] == ".tif"):
        im[:,:,i] = tf.imread(files_path + "/" + file)
        i += 1
elapsed = time.time() -start
        
print('Volume has size of ' + str(im.shape[0]) + ' by ' + str(im.shape[1]) + ' by ' + str(im.shape[2]))
print('Import time: ' + str(elapsed/60) + ' minutes')

#Volume_Downsample

start = time.time()

ds_factor = [round(max(voxel)/voxel[0]),round(max(voxel)/voxel[1]),round(max(voxel)/voxel[2])]

#im_d = ski.measure.block_reduce(im, block_size=ds_factor, func=np.max)

res = [round(im.shape[0] / ds_factor[0]), round(im.shape[1] / ds_factor[1]), round(im.shape[2] / ds_factor[2])]

im_d = ski.transform.resize(im, res, order=1, mode='reflect', cval=0, clip=True, preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None)

elapsed = time.time() - start
print('Downsample time: ' + str(elapsed/60) + ' minutes')
#Save Downsampled image:
tf.imsave(save_path + "01_downsampled.tif", np.moveaxis(im_d.astype(np.uint16), -1, 0))

print('Volume has size of ' + str(im_d.shape[0]) + ' by ' + str(im_d.shape[1]) + ' by ' + str(im_d.shape[2]))


#Spheroid Segmentation

#Segmentation - Guassian filter + intensity threshold
start = time.time()

im_g = gaussian_filter(im_d, sigma=gaussianblur)

elapsed = time.time() - start
print('Gaussian filter time: ' + str(elapsed/60) + ' minutes')
#Save Gaussian image:
tf.imsave(save_path + "02_gaussian.tif", np.moveaxis(im_g.astype(np.uint16), -1, 0))

start = time.time()

im_t = im_g > threshold #threshold based on intensity

elapsed = time.time() - start
print('Threshold time: ' + str(elapsed/60) + ' minutes')
#Save Thresholded image - as 8bit:
tf.imsave(save_path + "03_threshold.tif", np.moveaxis(im_t.astype(np.uint8),-1,0))

#Remove variables to make space for RAM
del(im_g)

#Find Thresholded locations
start = time.time()
im_l = ski.measure.label(im_t, connectivity=1)
rois = ski.measure.regionprops(im_l)
elapsed = time.time() -start
print('Labelling time: ' + str(elapsed/60) + ' minutes')

im_8 = (255*im_d/im_d.max()).astype(np.uint8) #make 8bit image for roi box
im_8_2 = np.copy(im_8)
start = time.time()
rois_f = [] #for filtered rois
for i, roi in enumerate(rois):
    if roi.area > 100:
        begin = (roi.bbox[0], roi.bbox[1])
        end = (roi.bbox[3]-1, roi.bbox[4]-1)
        xx,yy = ski.draw.rectangle_perimeter(begin, end=end, shape=im_8.shape, clip=True)
        im_8_2[xx,yy,roi.bbox[2]] = 255
        im_8_2[xx,yy,roi.bbox[5]-1] = 255
        for j in range(roi.bbox[2], roi.bbox[5]): #Good for z-step
            im_8[xx,yy,j] = 255
            im_8_2[roi.bbox[0],roi.bbox[1],j]=255
            im_8_2[roi.bbox[3]-1,roi.bbox[4]-1,j]=255
            im_8_2[roi.bbox[0],roi.bbox[4]-1,j]=255
            im_8_2[roi.bbox[3]-1,roi.bbox[1],j]=255
             
        rois_f.append(roi)

elapsed = time.time() - start
print('Roi drawing time: ' + str(elapsed/60) + ' minutes')
print('Number of ROIs filtered out = ' + str(len(rois)-len(rois_f)))

#Save bounding box image - as 8bit:
tf.imsave(save_path + "04_bboxes_sq.tif", np.moveaxis(im_8,-1,0))
tf.imsave(save_path + "04_bboxes_cu.tif", np.moveaxis(im_8_2,-1,0))
del(im_8)
del(im_8_2)

#Spherical_Cropping

#Save each filtered ROI:
start = time.time()
for i, roi in enumerate(rois_f):
    padded = np.array([(roi.bbox[0]-1)*ds_factor[0] - pad[0], (roi.bbox[1]-1)*ds_factor[1] - pad[1], (roi.bbox[2]-1)*ds_factor[2] - pad[2], (roi.bbox[3]+1)*ds_factor[0] + pad[0], (roi.bbox[4]+1)*ds_factor[1] + pad[1], (roi.bbox[5]+1)*ds_factor[2] + pad[2]])
    padded[0:3][padded[0:3] < 0] = 0
    padded[3]= padded[3] if padded[3] < im.shape[0] else im.shape[0]
    padded[4]= padded[4] if padded[4] < im.shape[1] else im.shape[1]
    padded[5]= padded[5] if padded[5] < im.shape[2] else im.shape[2]
    tf.imsave(save_path+"rois/roi_" + str(i) + ".tif", np.moveaxis(im[padded[0]:padded[3], padded[1]:padded[4], padded[2]:padded[5]].astype(np.uint16), -1, 0))
elapsed = time.time() -start
print('Saving time: ' + str(elapsed/60) + ' minutes')