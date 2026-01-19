# Created: Jan 10, 2026
# Last modified: Jan 10, 2026
# Last git commit: Jan 10, 2026
## Updates: Moved helper functions to locomotion_functions.py.

# Use this file to calculate the second-to-second movement of an individual. Will save as 
# Further binning should be done in a separate file. 

import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import os
import locomotion_functions as lf

# 1: Directories
dlc_dir = Path.cwd().parent
scripts_dir = dlc_dir.joinpath("Scripts")
results_dir = dlc_dir.joinpath("Results", "p14_oxtrko")
pose_dir = results_dir.joinpath("pose_estimation")
video_dir = dlc_dir.joinpath("Videos", "p14_isolation_cropped")
locomotion_dir = results_dir.joinpath("locomotion")
print(dlc_dir, scripts_dir, pose_dir, video_dir)

# 2: Create a list of csv files in the pose estimation directory, the names of those subjects, and their videos
csvs = lf.csv_list(pose_dir)
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
        x, y, p = lf.import_bodypart(csvs[i], part)
        
        sec_dists = lf.second_movement(x, y, fps, pix_per_cm_x, pix_per_cm_y)
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
