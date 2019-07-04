#___________________________________________________
#Imports

import math
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



#___________________________________________________
#Constants

voxel = [0.225,0.225,2] # x,y,z in um, specific to camera sensor and z steps
gaussianblur = [1, 1, 1] # px (after downsampling)
pad = [30, 30, 5] # px, padding in x, y, z, for bounding box 
bbox_zstack = False # show all squares (True) or wire frame (False)

threshold = 1200  # 16bit grey values
threshold2 = 2400  # 16bit grey value 2
threshold_edge = 25  # 8-bit grey value for edge thresholding to remove background

min_area = 40 # Max area for a singular spheroid
max_area = 140000 # Max area for a singular spheroid

shift_distance = 5 # TODO: calculate maximum shift_distance of ROI with voxel

ds_factor = [round(max(voxel)/voxel[0]),round(max(voxel)/voxel[1]),round(max(voxel)/voxel[2])] #factor for downscaling
# During downscaling, the boundingbox size factor becomes 1:1:1 for x, y, z



#___________________________________________________
#Function Declarations

def padding(obj_list): # apply padding to and conform ROIs to full-res, save location and resolution

    pad_list = [] # master list for padded ROIs LOCATIONS

    pad_sub = [] # location sublist 

    full_res_list = [] # master list for full_res ROI DIMENSION

    full_res_sub = [] # full-res dimension sublist

    for obj in range(len(obj_list)):
        
        for roi in range(len(obj_list[obj])):

            # conform rois to full resolution for cropping and apply padding

            padded = np.array([int((obj_list[obj][roi][0]-1)*ds_factor[0] - pad[0]), int((obj_list[obj][roi][1]-1)*ds_factor[1] - pad[1]), 
                        int((obj_list[obj][roi][2]-1)*ds_factor[2] - pad[2]), int((obj_list[obj][roi][3]+1)*ds_factor[0] + pad[0]), 
                        int((obj_list[obj][roi][4]+1)*ds_factor[1] + pad[1]), int((obj_list[obj][roi][5]+1)*ds_factor[2] + pad[2])])
            
            # in the event of an edge cut off, use bounding box edge

            padded[0:3][padded[0:3] < 0] = 0

            padded[3]= padded[3] if padded[3] < im.shape[0] else im.shape[0]
            padded[4]= padded[4] if padded[4] < im.shape[1] else im.shape[1]
            padded[5]= padded[5] if padded[5] < im.shape[2] else im.shape[2]

            # obtain roi dimension

            res5 = (padded[3] - padded[0], padded[4] - padded[1], padded[5] - padded[2])

            # append lists and empty sublists

            pad_sub.append(padded)
            full_res_sub.append(res5)
        
        pad_list.append(pad_sub)
        full_res_list.append(full_res_sub)

        pad_sub = []
        full_res_sub = []

    # use dictionary for dual funcion output

    d = dict()

    d['res'] = full_res_list
    d['dim'] = pad_list

    return d

def max_res (res5): # obtain maxium possible image dimension after conforming 
    
    max_0 = 0
    max_1 = 0
    max_2 = 0

    # maximum dimension on three axis

    for i in range(len(res5)):
        if res5[i][0] > max_0:
            max_0 = res5[i][0]
        if res5[i][1] > max_1:
            max_1 = res5[i][1]
        if res5[i][2] > max_2:
            max_2 = res5[i][2]

    return [max_0, max_1, max_2]


def save_rois_4d(pad, res5, start, num, files, filename): 
# use location and resolution data from padding() to import, crop and save ROIs into hyperstacks

    z = 0

    maxi = max_res(res5) # obtain maxium dimension
    im_out = np.zeros([maxi[0], maxi[1], maxi[2]*(frame-start)], dtype = np.uint16) # empty hyperstack
    print('ROI dimension',im_out.shape)

    for vol in range (num): # the number of frames the ROI spans

        for i , file in enumerate(files[(start+vol)*z_depth:len(files)- extra]): # search files excluding already saved files

            if (i >= pad[vol][2] and z < pad[vol][5]-pad[vol][2]): # use provided cropping window

                im_r = tf.imread(files_path + "/" + file) # read image files with temp variable im_r

                # Take the avalible portion of the image in the event of cut out
                im_out[0:res5[vol][0],0:res5[vol][1],vol*maxi[2] + z] = im_r[pad[vol][0]:pad[vol][0]+res5[vol][0], pad[vol][1]:pad[vol][1]+res5[vol][1]]
                z += 1
        z = 0 

    tf.imsave(rois_path + str(filename), np.moveaxis(im_out.astype(np.uint16), -1, 0))

    return


def save_rois (roi, files, count, exception): # unused 2d saving function, input == ('roi object'), img_in == ('2d arrays')

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

def save_rois_3d (roi, count, i, j, exception): # unused 2d saving function, input == ('roi object'), img_in == ('3d stacks')

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

def save_hyperstack (ndarray, method, filename, path): # convert a 4d array to a 3d hyperstack image

    res3 = ndarray.shape 

    im_out = np.zeros([res3[0], res3[1], int(res3[2]*res3[3])]) # empty image

    for i in range(res3[3]):
        im_out[:,:,(i*res3[2]):((i+1)*res3[2])]= ndarray[:,:,:,i] # stacking time axis on z_step

    if (method == 'uint16'):
        tf.imsave(path + str(filename), np.moveaxis(im_out.astype(np.uint16), -1, 0))
    elif (method == 'uint8'):
        im_out = (255*im_out/im_out.max()).astype(np.uint8)  # comform 16 bit to 8 bit output without overexposing
        tf.imsave(path + str(filename), np.moveaxis(im_out.astype(np.uint8), -1, 0))

    return

def draw_rois (bbox, greyscale, k):  # draw a bounding box on a 8 bit 4d array (configured as im_8) with greyscale value
    
    begin = (bbox[0], bbox[1])
    end = (bbox[3]-1, bbox[4]-1)

    # establish cubic diagonals 

    xx,yy = ski.draw.rectangle_perimeter(begin, end=end, shape=im_8[:,:,:,k].shape, clip=True)

    # draw lines on im_8_2

    im_8_2[xx,yy,bbox[2],k] = greyscale
    im_8_2[xx,yy,bbox[5]-1,k] = greyscale
    for j in range(bbox[2], bbox[5]): # draw along z_step
        im_8[xx,yy,j,k] = greyscale # draw lines on im_8

        # draw corner markers on im_8_2 
        im_8_2[bbox[0],bbox[1],j,k] = greyscale
        im_8_2[bbox[3]-1,bbox[4]-1,j,k] = greyscale
        im_8_2[bbox[0],bbox[4]-1,j,k] = greyscale
        im_8_2[bbox[3]-1,bbox[1],j,k] = greyscale

    return

def exception_advanced_threshold (ndarray, pad, dim, big_bbox, k): 

    # thresholding outer rim of ROIs to be easily seperateble by regionprops
    im_adv = ndarray - im_t[im_pad[0]:im_pad[3], im_pad[1]:im_pad[4], im_pad[2]:im_pad[5], k]*10 
    im_adv = gaussian_filter(im_adv, sigma=[5,5,1]) # diffuse the image for cleaner segmentation

    # label the boolean map with ski.measure

    im_label = ski.measure.label( threshold_edge < im_adv, connectivity = 1)
    rois_thr = ski.measure.regionprops(im_label)

    count = 0

    for j, roi in enumerate (rois_thr): 

            # size and edge constraints

        if (roi.area <= 70 or (roi.bbox[0] == 0 or roi.bbox[1] == 0 or roi.bbox[2] == 0 
            or roi.bbox[3] == im_d.shape[0] or roi.bbox[4] == im_d.shape[1] or roi.bbox[5] == im_d.shape[2])):
            continue
        else: 

            # create smaller sub-bounding box within the large exceptions

            final_bbox = [big_bbox[0]+roi.bbox[0], big_bbox[1]+roi.bbox[1], big_bbox[2]+ roi.bbox[2],
             big_bbox[3]-roi.bbox[3], big_bbox[4]-roi.bbox[4], big_bbox[5]-roi.bbox[5]]

            # Append all bbox into original list

            rois_frame.append(final_bbox)

            draw_rois(final_bbox, 255, k)
            # save_rois_3d (roi, count, i, j, True)
            count += 1

    return

def get_centroid (bbox): # return the centoid of a bounding box

    return [int((bbox[0]+bbox[3])/2), int((bbox[1]+bbox[4])/2), int((bbox[2]+bbox[5])/2)]

def get_distance (bbox1, bbox2): # return centroidal distance between two bounding boxes

    bbox1_c = get_centroid(bbox1)
    bbox2_c = get_centroid(bbox2)

    return math.sqrt((bbox2_c[0] - bbox1_c[0])**2 + (bbox2_c[1] - bbox1_c[1])**2 + (bbox2_c[2] - bbox1_c[2])**2)

def get_volume (bbox): # return volume of the bounding box

    return ((bbox[3]-bbox[0])*(bbox[4]-bbox[1])*(bbox[5]-bbox[2]))

def get_radius (bbox): # return radius/semi major axis of spheroid

    return max(int((bbox[3]-bbox[0])/2), int((bbox[3]-bbox[0])/2), int((bbox[3]-bbox[0])/2))

def get_res (bbox): # return bounding box resolution in three axis

    return [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]

def get_conformed_bbox (centroid_list, dim): # conform bounding box according to same box size on its centroid 

    conformed_list = []

    for i in range (len(centroid_list)):

        # add and subtract bounding box dimension on each centroid

        bbox = [centroid_list[i][0] - dim[0]/2, centroid_list[i][1] - dim[1]/2, centroid_list[i][2] - dim[2]/2,
            centroid_list[i][0] + dim[0]/2, centroid_list[i][1] + dim[0]/2, centroid_list[i][2] + dim[0]/2]

        conformed_list.append(bbox)

    return conformed_list




#___________________________________________________
#Import_Volume

root = tk.Tk()
root.withdraw()
files_path = askdirectory()
save_path = files_path + "/processed/"
os.mkdir(save_path)
rois_path = save_path + "/rois/"
os.mkdir(rois_path)


files = next(os.walk(files_path))[2] #all files within directory
for file in files:
    if (file[-4:] != ".tif"):
        files.remove(file) # remove any non-'.tif' file from the list

files.sort()

z_depth = 0

frame_num = 0

for i, file in enumerate(files):
    if (file[-13] != str(frame_num)): # if not counted to a new data set in filename yet
        z_depth = i # switch to next frame
        break
    frame_num = file[-13]

extra = len(files)%z_depth # remove extra z_stacks of a incomplete frame

frame = int((len(files)-extra)/z_depth) #calculate the number of frames

im = tf.imread(files_path + "/" + files[5])
im = np.zeros([im.shape[0], im.shape[1], z_depth], dtype = np.uint16) #make empty image
res = [round(im.shape[0] / ds_factor[0]), round(im.shape[1] / ds_factor[1]), round(im.shape[2] / ds_factor[2])]
im_d = np.zeros([res[0],res[1],res[2],frame]) # creating a 4d array

print('Original Volume has size of ' + str(im.shape[0]) + ' by ' + str(im.shape[1]) + ' by ' + str(im.shape[2]) + ' with',frame,'frames')

start = time.time()

# import 2d images to 4d arrays

j = 0

for i , file in enumerate(files[0:len(files)-extra]):
    if (i%z_depth == 0 and i != 0):
        j += 1
   
    im_r = tf.imread(files_path + "/" + file) # read image files with temp variable im_r
    im_d[:,:,(i%z_depth),j] = ski.transform.resize(im_r, res[0:2], order=1, mode='reflect', cval=0, 
                    clip=True, preserve_range=True, anti_aliasing=False, anti_aliasing_sigma=None) 
                    # Volume_Downsample and apply onto array

    if (i % 50 == 0):
            print("Importing and Downsampling files...", i,"/", len(files))

elapsed = time.time() -start        
print('Downsampled Volume has size of ' + str(im_d.shape[0]) + ' by ' + str(im_d.shape[1]) + ' by ' + str(im_d.shape[2]))
print('Import and Downsample time: ' + str(elapsed/60) + ' minutes')

save_hyperstack(im_d, 'uint16', '01_downsampled.tif', save_path)




#____________________________________________________
#Spheroid Segmentation

im_g = np.zeros_like(im_d, dtype = np.uint16)

start = time.time()

# Use Guassian filter to diffuse image

for i in range(frame):
    im_g[:,:,:,i] = gaussian_filter(im_d[:,:,:,i], sigma=gaussianblur)
   
elapsed = time.time() - start
print('Gaussian filter time: ' + str(elapsed/60) + ' minutes')

save_hyperstack(im_g, 'uint16', '02_gaussian.tif', save_path)

start = time.time()

# threshold based on intensity

im_t = im_g > threshold

elapsed = time.time() - start

print('Threshold time: ' + str(elapsed/60) + ' minutes')

save_hyperstack(im_t, 'uint8', '03_threshold.tif', save_path)

#Remove variables to make space for RAM
# del(im_g)



#____________________________________________________
# Populate ROIs and Identify Bounding Boxes


rois_bbox = [] # master list
rois_frame = [] # list of bbox(es) on one frame


im_8 = (255*im_d/im_d.max()).astype(np.uint8) #make 8bit image for roi box
im_8_2 = np.copy(im_8)

# start = time.time()
for k in range (frame):
    im_l = ski.measure.label(im_t[:,:,:,k], connectivity=1)
    rois = ski.measure.regionprops(im_l)
    elapsed = time.time() -start
    # print('Labelling time: ' + str(elapsed/60) + ' minutes')

    #print('\nNumber of ROIs detected = ', len(rois))

    start = time.time()

    rois_f = [] # for filtered rois
    rois_f2 = [] # for filtered exceptions

    # count = 0
    # count2 = 0

    rois_frame = []

    for i, roi in enumerate(rois):

        # size and edge restrictions 
        if (roi.area <= min_area or (roi.bbox[0] == 0 or roi.bbox[1] == 0 or roi.bbox[2] == 0 
            or roi.bbox[3] == im_d.shape[0] or roi.bbox[4] == im_d.shape[1] or roi.bbox[5] == im_d.shape[2])):
            continue

        if (roi.area > min_area and roi.area <= max_area): #threshold depends on avg size
            draw_rois(roi.bbox, 255, k)

            rois_f.append(roi)
            
            rois_frame.append(roi.bbox) # append this bbox into frame list
            

        elif (roi.area > max_area): # if size exceed threshold -> exception
        
            im_pad = ((roi.bbox[0]-1)-int(pad[0]/5), (roi.bbox[1]-1)-int(pad[1]/5), (roi.bbox[2]-1)-int(pad[2]/5),
            (roi.bbox[3]+1)+int(pad[0]/5),(roi.bbox[4]+1)+int(pad[1]/5),(roi.bbox[5]+1)+int(pad[2]/5))
            roi_res = (im_pad[3] - im_pad[0], im_pad[4] - im_pad[1], im_pad[5] - im_pad[2]) # crop roi

            #16 bit 3d image stack of exception ROI from threshold
            im_r3d = np.copy(np.uint16(im_g[im_pad[0]:im_pad[3], im_pad[1]:im_pad[4], im_pad[2]:im_pad[5], k])) 
            im_r3d = (im_r3d/im_r3d.max() * 255).astype(np.uint8)  # convert to 8 bit without overexposing

            rois_f2.append(roi)

            exception_advanced_threshold(im_r3d, im_pad, roi_res, roi.bbox, k) # apply futher crop with function

    rois_bbox.append(rois_frame) # append bbox(es) of this frame into the master list

elapsed = time.time() - start
print('ROI drawing and cropping time: ' + str(elapsed/60) + ' minutes')


#Save bounding box images - as 8bit:
save_hyperstack(im_8, 'uint8', '04_bboxes_sq.tif', save_path)
save_hyperstack(im_8_2, 'uint8', '04_bboxes_cu.tif', save_path)

del(im_8)
del(im_8_2)



#____________________________________________________
# Link ROIs among frames

start = time.time()

rois_bbox_copy = rois_bbox.copy

min_distance = math.sqrt(res[0]**2+res[1]**2) # Preset maximum distance (diagonal) across the downscaled image
min_roi = 0
rois_list = [] # Master list of linked bbox locations
rois_link = [] # list of linked ROIs of one object

start_frame = [] # list of the starting frame for each object

roi_break_off = False
            
for vol in range (frame): # Scan each frame for ROIs that does not span the whole z_depth (starts from a higher index)
    for r in range(len(rois_bbox[vol])): 
        rois_link.append(rois_bbox[vol][r]) # Always Start from the first ROI on frame == vol
        
        start_frame.append(vol) # write starting frame to the list

        # del(rois_bbox[vol][r])
        for f in range (frame-vol-1): # Iterate  throught the frames

            # print(f)
            min_d = min_distance # Reset maximun distance

            if (rois_bbox[f+1] == []): #break if next list is empty
                break

            for r_next in range(len(rois_bbox[f+1])): # The number of ROIs on the next frame of frame == f
                # print(r_next)
                
                if (min_d >= get_distance(rois_link[f], rois_bbox[f+1][r_next])): # ROIs on the next frame that is closest to rois_bbox[f][r] (rois_link[f]) 
                    # print(min_d, get_distance(rois_link[f], rois_bbox[f+1][r_next]))
                    min_roi = r_next
                    # print(min_roi,min_d,'\n')
                    min_d = get_distance(rois_link[f], rois_bbox[f+1][r_next])
                    # print(min_d)

            #_______________________________________
            # IN DEVELOPMENT
            #---------------------------------------

            #TODO: Implementing "look ahead" function where program will continue search ROIs with one ot more missing frame(s)



            # print(f, r, min_roi)

            # if (rois_bbox[f] == [] or get_radius(rois_bbox[f][min_roi])*1.2 <= get_distance(rois_link[f], rois_bbox[f+1][min_roi])):

            #     print(f,'\n')
                
            #     ahead_range = 4 if (f + 4) < frame else frame - f  #look ahead 4 frames without exceeding range

            #     for f_ahead in range (ahead_range): # special_cases

            #         print("1__",vol, f, r)

            #         if (rois_bbox[f_ahead + f] != []): # prevent index out of range, goto (special_case [2])
            #             print('case1')

            #             # for r_n in range(len(rois_bbox[f_ahead + f])): # (spacial_case [1]), lost of ROI on interframe
            #             #     if (min_d >= get_distance(rois_link[f], rois_bbox[f_ahead + f][r_n])): # ROIs on the next frame that is closest to rois_bbox[f][r] (rois_link[f]) 

            #             #         # print(get_radius(rois_bbox[f][min_roi])*1.2)
            #             #         # print(get_distance(rois_link[f], rois_bbox[f+1][min_roi]))
                                
            #             #         min_roi = r_n
            #             #         min_d = get_distance(rois_link[f], rois_bbox[f_ahead + f][r_n])

            #             #     if (get_radius(rois_bbox[f][min_roi])*1.2 >= get_distance(rois_link[f], rois_bbox[f_ahead + f][min_roi])):

            #             #         for f_a in range (f_ahead-1): # append ROIs as per last found bbox location for all instances before newly found ROI
            #             #             rois_link.append(rois_link[f])

            #             #         print("1__b", f, r_n, r)

            #             #         rois_link.append(rois_bbox[f_ahead + f][min_roi]) # then append newly found ROI
            #             #         del(rois_bbox[f_ahead + f][min_roi]) # delete instance in rois_bbox 
            #             #         roi_break_off_1 = True                          
            #             #         break # goto (breakpoint [1]) for continued search in existing ROI

            #         else: # (special_case [2]), empty ROI list on next frame

            #             print("case2__",vol, f, r)

            #             for f_ahead in range (ahead_range): # (special_case[2_1] -> special_case [1]), empty list due to lost of ROI on interframe
                            
            #                 if (rois_bbox [f_ahead + f] != []):
            #                     # rois_link.append(rois_link[f]) # append ROI as per last found bbox location for one instance
            #                     break # going into next iteration until a non_empty list feeds into interframe search

            #             roi_break_off_2 = True # (special_case[2_2], empty list due to ROI termination)

            #             # print("2__",vol, f, r)

            #             break # goto (breakpoint [2]) if the 4 frames aheads are empty  

            #         if(roi_break_off_1 == True): # (breakpoint [1]), search staring at existing ROI and new frame 
            #             print('breakpt1')
            #             break

            # elif(roi_break_off_2 == True):
            #     print('breakpt2')
            #     break # (breakpoint [2]), search starting new ROIs



            if (shift_distance >= get_distance(rois_link[f], rois_bbox[f+1][min_roi])): # general_case
            # shift distance restriction: the maximun allowed drift is calculated by voxel
                
                rois_link.append(rois_bbox[f+1][min_roi]) # Append choosen ROI

                del(rois_bbox[f+1][min_roi]) # delete instance in rois_bbox
        
        rois_list.append(rois_link) # Append list of ROIS into the master list
        rois_link = [] # clear list

elapsed = time.time() - start
print('ROI linking time: ' + str(elapsed/60) + ' minutes')



#____________________________________________________
# Saving ROIs into hyperstacks

# confrom rois_list onto a common resolution with padding and get centroid of rois

start = time.time()

# create empty lists for dimension and centroids

res_list = []
centroid_list = []

for obj in range(len(rois_list)):
    res4 = np.zeros(3, dtype = np.int32) # empty 3d numpy array

    for roi in range(len(rois_list[obj])):
        res4 += get_res(rois_list[obj][roi]) # addition of dimenison that leads to an average
        centroid_list.append(get_centroid(rois_list[obj][roi])) # append centroids to a list

    res4 = (res4 / len(rois_list[obj])) # averaging bbox size 
    res_list.append(get_conformed_bbox(centroid_list, res4)) # conform bboxes with cnetroids and new dimensions
    centroid_list = []


d = padding(res_list)

full_res_list = d['res']
pad_list = d['dim']

# Import images and creating 3d numpy arrays of ROIs

for obj in range (len(pad_list)):
    # import and save hyper stacks with designated namings
    obj_4d = save_rois_4d(pad_list[obj], full_res_list[obj], start_frame[obj], len(rois_list[obj]), files, str("roi_" + str(obj) + ".tif")) # croped 4d numpy array for each object
    print("Saving ROIs...", obj+1,"/", len(pad_list))


elapsed = time.time() - start
print('ROI Saving time: ' + str(elapsed/60) + ' minutes')




#______________________________________________________
# End of the program





#_______________________________________
# IN DEVELOPMENT
#---------------------------------------

#TODO: Implementing ROI detection with opencv's cv2.HoughCircles() method 




# im_c = np.full((len(rois_f2), 200, 10, 3), 0) #arbitary max z-size = 200
# im_r3d = (im_r3d / im_r3d.max() * 180 + im_t * 60).astype(np.uint8)
# im_r3d = gaussian_filter(im_r3d, sigma=[2,2,2])

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