# Created: Jan 10, 2026
# Last modified: Jan 10, 2026
# Last git commit: Jan 20, 2026
## Updates: migrated from original analyze_locomotion.py file to better isolate functions.

# Use this file to calculate the second-to-second movement of an individual. 
# Further binning should be done in a separate file. 

import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import os

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


# 1: Directories
dlc_dir = Path.cwd().parent
scripts_dir = dlc_dir.joinpath("Scripts")
results_dir = dlc_dir.joinpath("Results", "p14_oxtrko")
pose_dir = results_dir.joinpath("pose_estimation")
video_dir = dlc_dir.joinpath("Videos", "p14_isolation_cropped")
locomotion_dir = results_dir.joinpath("locomotion")
print(dlc_dir, scripts_dir, pose_dir, video_dir)

# 2: Create a list of csv files in the pose estimation directory, the names of those subjects, and their videos
csvs = csv_list(pose_dir)
## testing: only use first individual in list
# csvs = csvs[0:1]
names = [csv.name.split("p14_isolation_")[1].split("DLC")[0] for csv in csvs] # check that this line works in HPG. likely a better way
videos = [video_dir.joinpath("p14_isolation_"+name+".mp4") for name in names]

bodyparts = ["nose", "head_midpoint", "mouse_center", "tail_base", "tail3", "tail_end", "left_shoulder", "right_shoulder", "left_midside", "right_midside", "left_hip", "right_hip"]
fps = 15

# 3: Create a multi-indexed pandas array with bodypart and movement for each second; export to csv

for i, ind in enumerate(names):
    dfs_sec = []
    cup_coord_filename = Path(f"{dlc_dir}/Results/p14_oxtrko/locomotion/roi_coordinates/{ind}_cup_coords.pkl")
    if cup_coord_filename.is_file():
        print(f"{ind}'s video has already been labeled.")
    else:
        print(f"{ind}'s video has not had an ROI labeled. Please use label_cups.py to label it.")
        continue

    with open(cup_coord_filename, 'rb') as f: 
        roi_coord_data = pickle.load(f)
    x1 = roi_coord_data["x1"]
    x2 = roi_coord_data["x2"]
    y1 = roi_coord_data["y1"]
    y2 = roi_coord_data["y2"]
    pix_per_cm_x = roi_coord_data["pix_per_cm_x"]
    pix_per_cm_y = roi_coord_data["pix_per_cm_y"]

    for part in bodyparts:
        x, y, p = import_bodypart(csvs[i], part)
        
        sec_dists = second_movement(x, y, fps, pix_per_cm_x, pix_per_cm_y)
        n = len(sec_dists)
        df = pd.DataFrame(
            {"distance": sec_dists},
            index=pd.MultiIndex.from_arrays(
                [
                    np.repeat(ind, n),
                    np.repeat(part, n),
                    np.arange(n)
                ],
                names=["name", "bodypart", "second"]
            )
        )
        dfs_sec.append(df)

    dfs_sec = pd.concat(dfs_sec)
    dfs_sec.to_csv(Path(f"{locomotion_dir}/second_movement/{ind}_sec_dists.csv"))
