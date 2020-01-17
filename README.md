# Spheroid_Analysis
Code to import, downsample, crop SPIM images, filter ROIs, and generate bounding box for individual spheroid.

Created in May 2019, version 1.1 updated in October 2019                                                
First Implementation by Aaron Au, continued development by Ziyang Yu

Features and Improvements in this version:
1) 4x import and downsample speedup by using line-skipping                                                        
2) Implemented numba acceleration to some mathmatical functions                                                  
3) Implemented tqdm loading bar for execution progress visualization                                              
4) Added ability to save and import bbox file in .npy                                                           
5) Added ability to save bbox data file with saved crops in .txt                                               
6) Refined crop size estimation                                                                                  
7) Fixed index out of range error during large dataset processing                                                
8) Rudimentary implementation of "look_ahead"                                                                  
9) Other performance and stability improvements                                                                 

This is a stable version. Achieves ~85% successful tracking on SPIM 30 min interval data sets

Version 2.0 will be a major feature update with new, Pandas and DL powered backend.
