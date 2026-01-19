# Created: Jan 19, 2026
# Last modified: Jan 19, 2026
# Last git commit: 
## Updates: created. 

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

def select_roi(vid_path, message):
    # cup dimater: 7.5 cm
    x_diam = 7.5
    y_diam = 7.5
    
    # pixels-to-centimeters conversion
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {vid_path}")
    print("video opened")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

    success, frame = cap.read()

    if success: 
        print(f"Successfully captured frame")
    else:
        print(f"Error: could not read frame.")

    cap.release()

    window_name = message
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allows resizing
    cv2.resizeWindow(window_name, 1280, 720) # Set to 1280x720 pixels
    x1, y1, x2, y2 = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)

    width = x2 - x1
    height = y2 - y1
    # convert pixels to cm 
    pix_per_cm_x = width/x_diam
    pix_per_cm_y = height/y_diam
    
    return x1, y1, x2, y2, pix_per_cm_x, pix_per_cm_y


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
    df = pd.read_csv(f, skiprows=1, header=0)   # skip first row, make headers bodyparts (new row 0)
    df = df.filter(like=bodypart, axis=1)[1:]   # filter for rows containing 'bodypart' (pandas will import as e.g. nose, nose.1, nose.2)
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