#Imports
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
import cv2

#Function definitions
def padding (roi):
    padded = np.array([(roi.bbox[0]-1)*ds_factor[0] - pad[0], (roi.bbox[1]-1)*ds_factor[1] - pad[1], (roi.bbox[2]-1)*ds_factor[2] - pad[2], (roi.bbox[3]+1)*ds_factor[0] + pad[0], (roi.bbox[4]+1)*ds_factor[1] + pad[1], (roi.bbox[5]+1)*ds_factor[2] + pad[2]])
    padded[0:3][padded[0:3] < 0] = 0
    padded[3]= padded[3] if padded[3] < im.shape[0] else im.shape[0]
    padded[4]= padded[4] if padded[4] < im.shape[1] else im.shape[1]
    padded[5]= padded[5] if padded[5] < im.shape[2] else im.shape[2]

    res2 = (padded[3] - padded[0], padded[4] - padded[1], padded[5] - padded[2])
    d = dict()
    d['res'] = res2
    d['dim'] = (padded[0], padded[1], padded[2])

    return d

def save_rois (roi, files, count, exception):

    d = padding(roi)
    
    res2 = d['res']
    dim = d['dim']

    im_o = np.zeros(res2) # copy dimensions

    k = 0

    for file in files[dim[2]:dim[2]+res2[2]]:
        if (file[-4:] == ".tif"):
            im_r = tf.imread(files_path + "/" + file) # read image files with temp variable im_r
            im_o[:,:,k] = im_r[dim[0]:dim[0]+res2[0], dim[1]:dim[1]+res2[1]] # save cropped area
            k +=1

    if (exception == False):
        tf.imsave(save_path+"rois/roi_" + str(count) + ".tif", np.moveaxis(im_o.astype(np.uint16), -1, 0))
    else:
        tf.imsave(save_path+"rois/roi_" + str(count) + "_exception"+ ".tif", np.moveaxis(im_o.astype(np.uint16), -1, 0)) # exception raised in filename
   
    return

def save_rois_3d (roi, count, i, j, exception):

    exp_files = next(os.walk(save_path+"rois/"))[2] #all files within directory

    d = padding(roi)
    
    res2 = d['res']
    dim = d['dim']

    im_o = np.zeros(res2) # copy dimensions

    for file in exp_files:
        if (file[-13:] == "exception.tif"):
            if (file[-15] == str(i)):
                im_read = np.moveaxis(tf.imread(save_path + "/rois/" + file),0 ,-1) # read image files with temp variable im_r
                im_o = im_read[dim[0]:dim[0]+res2[0], dim[1]:dim[1]+res2[1],dim[2]:dim[2]+res2[2]] # save cropped area

    tf.imsave(save_path+"rois/roi_" + str(count) + "_reprocessed_"+str(i)+'-'+str(j)+ ".tif", np.moveaxis(im_o.astype(np.uint16), -1, 0)) # exception raised in filename
    
    return

#Constants

voxel = [0.225,0.225,2] #x,y,z in um

gaussianblur = [1, 1, 1] #px (after downsampling)
threshold = 1200 #grey values
threshold2 = 2400  #grey value 2
max_area = 140000 #Max area for a singular spheroid
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
os.mkdir(save_path+"rois/"+"exceptions/")


files = next(os.walk(files_path))[2] #all files within directory
for file in files:
    if (file[-4:] != ".tif"):
        files.remove(file) # remove any non-.tif file from the list

files.sort()

im = tf.imread(files_path + "/" + files[5])
im = np.zeros([im.shape[0], im.shape[1], len(files)], dtype = np.uint16) #make empty image
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
count2 = 0

for i, roi in enumerate(rois):
    if (roi.area <= 70 or (roi.bbox[0] == 0 or roi.bbox[1] == 0 or roi.bbox[2] == 0 or roi.bbox[3] == im_d.shape[0] or roi.bbox[4] == im_d.shape[1] or roi.bbox[5] == im_d.shape[2])):
        continue
    if (count % 10 == 0):
        print('Saving ROIs...', count,"/", len(rois))
    if (roi.area > 70 and roi.area <= max_area): #threshold depends on avg size
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

    elif (roi.area > max_area):
        
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

        save_rois(roi, files, count2, True)
        count2 += 1 
        rois_f2.append(roi)

# catalorizing

     

elapsed = time.time() - start
print('ROI drawing and cropping time: ' + str(elapsed/60) + ' minutes')
print('Number of ROIs with size exceeding threshold = ', len(rois_f2))
print('Number of ROIs filtered out = ' + str(len(rois)-len(rois_f)-len(rois_f2)))

#Save bounding box image - as 8bit:
tf.imsave(save_path + "04_bboxes_sq.tif", np.moveaxis(im_8,-1,0))
tf.imsave(save_path + "04_bboxes_cu.tif", np.moveaxis(im_8_2,-1,0))
del(im_8)
del(im_8_2)

# Process exceptions

# im_c = np.full((len(rois_f2), 200, 10, 3), 0) #arbitary max z-size = 200


# im_r3d = (im_r3d / im_r3d.max() * 180 + im_t * 60).astype(np.uint8)

# im_r3d = gaussian_filter(im_r3d, sigma=[2,2,2])

count = 0

for i, roi in enumerate(rois_f2): 

    print('Processing exceptions... ', i+1)

    im_pad = ((roi.bbox[0]-1)-int(pad[0]/5), (roi.bbox[1]-1)-int(pad[1]/5), (roi.bbox[2]-1)-int(pad[2]/5),
    (roi.bbox[3]+1)+int(pad[0]/5),(roi.bbox[4]+1)+int(pad[1]/5),(roi.bbox[5]+1)+int(pad[2]/5))        # read image files with temp variable im_r
    res3 = (im_pad[3] - im_pad[0], im_pad[4] - im_pad[1], im_pad[5] - im_pad[2])

    im_r3d = np.copy(np.uint16(im_g[im_pad[0]:im_pad[3], im_pad[1]:im_pad[4], im_pad[2]:im_pad[5]])) #16 bit 3d image stack of exception ROI from threshold

    im_r3d = (im_r3d/im_r3d.max() * 255).astype(np.uint8)  # convert to 8 bit without overexposing

    # ____________________________________________
    #roi_f2 advanced_threshold_block

    im_adv = im_r3d - im_t[im_pad[0]:im_pad[3], im_pad[1]:im_pad[4], im_pad[2]:im_pad[5]]*10 #thresholding outer rim of ROIs
    im_adv = gaussian_filter(im_adv, sigma=[5,5,1])
    loc_bool = 25 < im_adv #arbitary 8-bit grey value for edge thresholding to remove background

    tf.imsave(save_path +"rois/exceptions/"+ 'roi_'+ str(i+1)+'threshold.tif',np.moveaxis(loc_bool.astype(np.uint8), -1, 0))

    im_l = ski.measure.label(loc_bool, connectivity=1)
    rois_thr = ski.measure.regionprops(im_l)

    for j, roi in enumerate (rois_thr):
        if (roi.area <= 70 or (roi.bbox[0] == 0 or roi.bbox[1] == 0 or roi.bbox[2] == 0 or roi.bbox[3] == im_d.shape[0] or roi.bbox[4] == im_d.shape[1] or roi.bbox[5] == im_d.shape[2])):
            continue
        else: 
            save_rois_3d (roi, count, i, j, True)
            count += 1

print(count,'new ROIs are saved... ')
 # del rois_f2[i]

    # ____________________________________________
    #roi_f2 opencv block(method = opencv)
    

    # for j in range (res3[2]):
    #     im_r2d = im_r3d[:,:,j] # convert to 2d for Hough detection 
    #     circles = cv2.HoughCircles(im_r2d, cv2.HOUGH_GRADIENT, 1, 30,
    #                   param1=1,
    #                   param2=1,
    #                   minRadius=15,
    #                   maxRadius=30)
    #     #print (circles.shape)
    #     circles = np.around(circles)
    #     for c in circles[0,:]:
    #         # cv2.circle(tuple(im_r2d),(k[0],k[1]),k[2],cir(125),2)ci
    #         # cv2.circle(tuple(im_r2d),(k[1]),2,(255),3)
    #         if c[0] < im_r2d.shape[0] and c[1] < im_r2d.shape[1] and c[0]+c[2] < im_r2d.shape[0] and c[1]+c[2] < im_r2d.shape[1] and  c[0]-c[2] > 0  and c[1]-c[2] > 0:   
    #             im_r2d[int(c[0]),int(c[1])] = 200
    #             xx,yy = ski.draw.rectangle_perimeter((c[0]-c[2], c[1]-c[2]), end = (c[0]+c[2],c[1]+c[2]), shape=im_r2d.shape, clip=True)
    #             im_r2d[xx,yy] = 200
    #             # im_c[i][j][k] = c
    #             # im_r3d[xx,yy,roi.bbox[5]-1] = 255
    #              # tf.imsave(save_path +"rois/"+ 'roi_'+ str(i+1) + '_'+ str(k[2])+'_'+str(k[0])+'_'+str(k[1])+'.jpg', im_r2d.astype(np.uint8))
    #     tf.imsave(save_path +"rois/"+ 'roi_'+ str(i+1) + '_' + str(j) +'.tif', im_r2d.astype(np.uint8))