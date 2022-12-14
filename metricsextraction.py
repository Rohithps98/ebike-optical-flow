import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math


import folium
import pyproj
from pyproj import Proj, CRS, transform

from geopy.distance import geodesic
import geographiclib
from geographiclib.geodesic import Geodesic



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime,timedelta

import pickle
import collections

import scipy
from scipy import signal, spatial

import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

import sys

#sys.path.append('C://Users/MaxGr/Desktop/Python/PII/GPSMapping')

# import metricExtract.py


Project_PATH = os.path.dirname(os.path.abspath('/Users/rohith/Desktop/thesis/videos/video3.mov'))


def get_target_i(dataframe, target_index):
    '''
    Get target i from all frames

    '''

    target_i_index = np.where(dataframe[:, index_Target] == target_index)[0]
    target_i = dataframe[target_i_index]

    return target_i, target_i_index


def get_frame_i(dataframe, frame_index):
    '''
    Get frame i from all frames

    '''

    frame_i_index = np.where(dataframe[:, index_Frame] == frame_index)[0]
    frame_i = dataframe[frame_i_index]

    return frame_i, frame_i_index


'''
Set index for the extraced data
'''
index_Time = 0
index_Frame = 1
index_Target = 2
index_x = 3
index_y = 4
index_class = 5
index_FrameQuantity = 6
index_Mean_V = 7
index_Instantaneous_V = 8
index_Traj_pred_score = 9  # Not used in our paper
index_TTC = 10


def get_TTC(dataframe):
    FRAME = np.unique(dataframe[:, index_Frame])

    TTC_FRAME = np.zeros((len(FRAME), 2))
    TTC_FRAME[:, 0] = FRAME

    TTC = [[], []]

    dense_map = np.zeros((1500, 500), dtype=np.uint8)
    dense_unit = 60  # 100 v12 for test 1

    for f in range(len(FRAME)):
        if (f % 30) == 0:  # print(f)

            frame, frame_list = get_frame_i(dataframe, FRAME[f])
            num_vehicle = len(frame)

            if num_vehicle == 0:
                continue

            # distance-based clusters
            for n in range(frame.shape[0]):
                i = round(frame[n, 4])
                j = round(frame[n, 3])

                for x in range(2 * dense_unit + 1):
                    for y in range(2 * dense_unit + 1):
                        if (x - dense_unit) ** 2 + (y - dense_unit) ** 2 <= dense_unit ** 2:
                            if i - dense_unit + x > 0 and i - dense_unit + x < 1500 and j - dense_unit + y > 0 and j - dense_unit + y < 500:
                                dense_map[i - dense_unit + x, j - dense_unit + y] = 255

            num_objects, labels = cv2.connectedComponents(dense_map)

            # if num_objects == 1:
            #     flag_1 = 1

            # if num_objects == 2:
            #     flag_2 = 1
            # Cat Cluster
            # 0. cluster
            # 1. x
            # 2. y
            # 3. V
            # 4.
            # 5.
            # 6. ttc
            cluster = np.zeros((num_objects - 1, 7), dtype=object)
            TTC_i = []

            if num_objects < 3:
                TTC_i = 0
                TTC[0].append(f)
                TTC[1].append(TTC_i)

            else:
                # Get cluster
                for i in range(1, num_objects):
                    cluster[i - 1, 0] = i

                    cluster_i_x = np.where(labels == i)[1]
                    cluster_i_y = np.where(labels == i)[0]
                    x_max = np.max(cluster_i_x)
                    x_min = np.min(cluster_i_x)
                    y_max = np.max(cluster_i_y)
                    y_min = np.min(cluster_i_y)

                    # cluster_i_x = (x_max + x_min)//2
                    # cluster_i_y = (y_max + y_min)//2
                    cluster_i_x = np.mean(cluster_i_x)
                    cluster_i_y = np.mean(cluster_i_y)

                    # print(cluster_i_y)
                    cluster[i - 1, 1] = cluster_i_x
                    cluster[i - 1, 2] = cluster_i_y

                    Vt_mean = []
                    for n in range(len(frame)):
                        if frame[n, 4] > y_min and frame[n, 4] < y_max:
                            Vt_mean.append(frame[n, 8])

                    cluster_Vmean = np.mean(Vt_mean)
                    cluster[i - 1, 3] = cluster_Vmean

                max_y = 900  # THE potential collision point predefine for segment 2
                # Get TTC
                for i in range(len(cluster)):

                    cluster_i = cluster[i]
                    # if cluster_i[2]>max_y:
                    #     break

                    # Caclute TTC for segment#1
                    # cluster_sub = np.delete(cluster, i, 0)

                    # #dist = np.where(abs(cluster_sub[:,1]-cluster_i[1]) > 60)
                    # dist = np.where(cluster_sub[:,2] > max_y )[0]
                    # minus_dist = np.where(cluster_sub[:,2]-cluster_i[2] <0 )[0]   ## LOOK FORWARD CLUSTER
                    # # print(dist)
                    # # print(minus_dist)
                    # dist = list(set(dist)|set(minus_dist))
                    # # print(dist)

                    # cluster_sub = np.delete(cluster_sub, dist, 0)
                    # # print(len(cluster_sub))
                    # # #cluster_sub = np.delete(cluster_sub, minus_dist, 0)
                    # # print(len(cluster_sub))
                    # if len(cluster_sub) == 0:
                    #     #TTC_i.append(0)   # 0?
                    #     pass
                    if True:
                        # dist_y = cluster_sub[:,2]
                        # cluster_n = cluster_sub[dist_y==min(dist_y)][0]
                        # ttc = (cluster_i[2]-cluster_n[2])/10 / (cluster_i[3]-cluster_n[3])
                        ttc = ((max_y - cluster_i[2]) / 10 / cluster_i[3])  # ttc for segment 2 . the merge .
                        # if ttc<0:
                        # print(False)
                        # print(cluster_i[2], cluster_n[2], cluster_i[3], cluster_n[3], ttc)

                        # if ttc < np.inf and ttc >= 0:
                        #     TTC_i.append(ttc)
                        # else:
                        #     TTC_i.append(nan)

                        if ttc < np.inf and ttc >= 0:
                            TTC_i.append(ttc)

                if len(TTC_i) > 0:  # get TTC-CV
                    TTC_i = np.nanstd(TTC_i) / np.nanmean(TTC_i)
                    TTC_i = TTC_i / ((num_objects - 1) / num_vehicle)
                    if np.isnan(TTC_i):
                        TTC_i = 0
                    TTC[0].append(f)
                    TTC[1].append(TTC_i)

                else:
                    print(TTC_i)

                # if np.isnan(TTC_i) :
                #     TTC_i = 0
                # else:
                #     TTC_i = TTC_i/((num_objects-1)/num_vehicle)

                # TTC[0].append(f)
                # TTC[1].append(TTC_i)

                print(f, TTC_i)

    # print(len(TTC[1]))
    TTC = np.nanmean(TTC[1])  # * 1000
    print(TTC)

    return TTC