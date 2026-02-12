# Created: Jan 19, 2026
# Last modified: Jan 19, 2026
# Last git commit: Jan 19, 2026
## Updates: created. Added csv_list, select_roi, import_bodypart, second_movement.

# Functions used for locomotion analysis with DLC pose estimation data. 

# typical import: import locomtion_functions as lf 

import cv2
import os
import numpy as np
from pathlib import Path
import pandas as pd

def csv_list(dir):
    """
    Creates list of csv files in a given directory. 
    """
    csvs = []
    for filename in os.listdir(dir):
        if filename.endswith((".csv")): 
            csvs.append(Path(os.path.join(dir, filename)))
    
    return csvs




def import_bodypart(f, bodypart):
    """
    imports coordinates and likelihoods for a specific bodypart from pose estimation csv file
    input: 
        f: csv file containing pose estimation coordinates
        bodypart: a string with the name of the body part to import 
    output: 
        x: array of bodypart's x coordinates from all frames
        y: array of bodypart's y coordinates from all frames
        p: array of bodypart's likelihood values from all frames
    """
    df = pd.read_csv(f, header=[0,1,2])   # skip first row, make headers bodyparts (new row 0)
    df = df.filter(like=bodypart, axis=1)   # filter for rows containing 'bodypart' (pandas will import as e.g. nose, nose.1, nose.2)
    # print(df.head())
    
    x = np.array(df.iloc[:, 0], dtype=float)
    y = np.array(df.iloc[:, 1], dtype=float)
    p = np.array(df.iloc[:, 2], dtype=float)

    return x, y, p


def second_movement(x, y, fps, pix_per_cm_x, pix_per_cm_y):
    """
    Calculates second-to-second movement of a single body part (e.g. for use in plotting). 
    inputs: 
        x: x coordinates of body part
        y: y coordinates of body part
        fps: frames per second of video
    outputs: 
        dists: an array of velocity from second-to-second (e.g. for use in plotting)
    """
    x_secs = x[::fps] # x position at end of every second
    y_secs = y[::fps] # y position at end of every second 

    dists = np.zeros(len(x_secs))

    # calculate cm traveled every second using euclidean distance formular and pix per cm conversion factor 
    for i in range(1, len(x_secs)):
        dists[i] = np.sqrt(( (x_secs[i] - x_secs[i-1])/pix_per_cm_x )**2 + ( (y_secs[i] - y_secs[i-1])/pix_per_cm_y )**2)

    return dists

def binned_dist(sec_dists, time, unit):
    """
    Function that returns the total distance moved over a given period of time
    Inputs: 
        sec_dists: array with total distance moved per second (i.e. the output of second_movement)
        time: period of time over which to chunk
        unit: unit to go with time; 's' = sec, 'm' = min
            e.g. to do 5 minute chunks, time = 5, unit = 'm'
    Outputs: 
        dists: total distance traveled per chunk of time 
    """ 
    if unit.lower() == 's':
        n = 1
    elif unit.lower() == 'm':
        n = 60
    else:
        raise ValueError("unit input should be 's' or 'm'")

    bin_size = n * time
    sec_dists = np.asarray(sec_dists)

    n_full_bins = len(sec_dists) // bin_size
    sec_dists = sec_dists[:n_full_bins * bin_size]

    return sec_dists.reshape(n_full_bins, bin_size).sum(axis=1)