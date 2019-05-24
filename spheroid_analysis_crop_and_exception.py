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

# Function definitions

def save_rois (roi, files, count, exception):
    padded = np.array([(roi.bbox[0]-1)*ds_factor[0] - pad[0], (roi.bbox[1]-1)*ds_factor[1] - pad[1], (roi.bbox[2]-1)*ds_factor[2] - pad[2], (roi.bbox[3]+1)*ds_factor[0] + pad[0], (roi.bbox[4]+1)*ds_factor[1] + pad[1], (roi.bbox[5]+1)*ds_factor[2] + pad[2]])
    padded[0:3][padded[0:3] < 0] = 0
    padded[3]= padded[3] if padded[3] < im.shape[0] else im.shape[0]
    padded[4]= padded[4] if padded[4] < im.shape[1] else im.shape[1]
    padded[5]= padded[5] if padded[5] < im.shape[2] else im.shape[2]

    res2 = (padded[3] - padded[0], padded[4] - padded[1], padded[5] - padded[2]) # define cropped 2d image dimensions

    im_o = np.zeros(res2) # copy dimensions

    k = 0

    for file in files[padded[2]:padded[5]]:
        if (file[-4:] == ".tif"):
            im_r = tf.imread(files_path + "/" + file) # read image files with temp variable im_r
            im_o[:,:,k] = im_r[padded[0]:padded[3], padded[1]:padded[4]] # save cropped area
            k += 1

    if (exception == False):
        tf.imsave(save_path+"rois/roi_" + str(count) + ".tif", np.moveaxis(im_o.astype(np.uint16), -1, 0))
    else:
        tf.imsave(save_path+"rois/roi_" + str(count) + "_exception"+ ".tif", np.moveaxis(im_o.astype(np.uint16), -1, 0)) # exception raised in filename

    return


#Constants

voxel = [0.225,0.225,2] #x,y,z in um

gaussianblur = [1, 1, 1] #px (after downsampling)
threshold = 1200 #grey values
threshold2 = 2400  #grey value 2
bbox_zstack = False # show all squares (True) or wire frame (False)
pad = [30, 30, 5] #px, padding in x, y, z, for bounding box 

ds_factor = [round(max(voxel)/voxel[0]),round(max(voxel)/voxel[1]),round(max(voxel)/voxel[2])] #factor for downscaling
# During downscaling, the boundingbox size factor becomes 1:1:1 for x, y, z

#Import_Volume
root = tk.Tk()
root.withdraw()
files_path = askdirectory()
save_path = files_path + "/processed/"
os.mkdir(save_path)
os.mkdir(save_path+"rois/")

files = next(os.walk(files_path))[2] #all files within directory
files.sort()

im = tf.imread(files_path + "/" + files[5])
im = np.zeros([im.shape[0], im.shape[1], len(files)]) #make empty image
res = [round(im.shape[0] / ds_factor[0]), round(im.shape[1] / ds_factor[1]), round(im.shape[2] / ds_factor[2])]
im_d = np.zeros(res)

print('Original Volume has size of ' + str(im.shape[0]) + ' by ' + str(im.shape[1]) + ' by ' + str(im.shape[2]))
         
i = 0

start = time.time()


for file in files:
    if (file[-4:] == ".tif"):
        # read image files with temp variable im_r
        im_r = tf.imread(files_path + "/" + file)

        #Volume_Downsample
        im_d[:,:,i] = ski.transform.resize(im_r, res[0:2], order=1, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None)

        if (i % 20 == 0):
            print("Importing and Downsampling files...", i,"/", len(files))
        i += 1

elapsed = time.time() -start        
print('Downsampled Volume has size of ' + str(im_d.shape[0]) + ' by ' + str(im_d.shape[1]) + ' by ' + str(im_d.shape[2]))
print('Import and Downsample time: ' + str(elapsed/60) + ' minutes')

tf.imsave(save_path + "01_downsampled.tif", np.moveaxis(im_d.astype(np.uint16), -1, 0))

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
# del(im_g)

#Find Thresholded locations
start = time.time()
im_l = ski.measure.label(im_t, connectivity=1)
rois = ski.measure.regionprops(im_l)
elapsed = time.time() -start
print('Labelling time: ' + str(elapsed/60) + ' minutes')
print('Number of ROIs detected = ', len(rois))

im_8 = (255*im_d/im_d.max()).astype(np.uint8) #make 8bit image for roi box
im_8_2 = np.copy(im_8)

start = time.time()

rois_f = [] #for filtered rois
rois_f2 = []

count = 0

for i, roi in enumerate(rois):
    if (count % 10 == 0):
        print('Saving ROIs...', count,"/", len(rois))
    if (roi.area > 70 and roi.area <= 140000): #threshold depends on avg size
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

        save_rois(roi, files, count, False)
        count += 1
        rois_f.append(roi)

    elif (roi.area > 140000):
        
        # print(i,'\n')
        # print(roi.centroid,'\n')
        # print(roi.bbox,'\n')
        # print(roi.area,'\n') # roi.area sum up every labeled pixel in each frame marked by roi

        #draw grey boxes for rois with size exceeding threshold
        begin = (roi.bbox[0], roi.bbox[1])
        end = (roi.bbox[3]-1, roi.bbox[4]-1)
        xx,yy = ski.draw.rectangle_perimeter(begin, end=end, shape=im_8.shape, clip=True)
        im_8_2[xx,yy,roi.bbox[2]] = 100
        im_8_2[xx,yy,roi.bbox[5]-1] = 100
        for j in range(roi.bbox[2], roi.bbox[5]): #Good for z-step
            im_8[xx,yy,j] = 100
            im_8_2[roi.bbox[0],roi.bbox[1],j]=100
            im_8_2[roi.bbox[3]-1,roi.bbox[4]-1,j]=100
            im_8_2[roi.bbox[0],roi.bbox[4]-1,j]=100

        save_rois(roi, files, count, True)
        count += 1 
        rois_f2.append(roi)
     

elapsed = time.time() - start
print('ROI drawing and cropping time: ' + str(elapsed/60) + ' minutes')
print('Number of ROIs with size exceeding threshold = ', len(rois_f2))
print('Number of ROIs filtered out = ' + str(len(rois)-len(rois_f)-len(rois_f2)))

#Save bounding box image - as 8bit:
tf.imsave(save_path + "04_bboxes_sq.tif", np.moveaxis(im_8,-1,0))
tf.imsave(save_path + "04_bboxes_cu.tif", np.moveaxis(im_8_2,-1,0))
del(im_8)
del(im_8_2)