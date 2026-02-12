# Created: Jan 10 2026
# Last modified: Jan 19 2026
# Last git commit: Jan 19 2026
## Updates: Moved helper functions to locomotion_functions.py

# Use this file to label the region of interest (ROI) for your experiment. 
# Jan 2026: For p14 sensory oxtrko isolation: highlighting the bottom of the cup. 

import cv2
import os
from pathlib import Path
import pickle
import locomotion_functions as lf 
import pandas as pd

# 1: Directories
dlc_dir = Path.cwd().parent.parent
scripts_dir = dlc_dir.joinpath("Scripts")
results_dir = dlc_dir.joinpath("Results", "p14_oxtrko")
pose_dir = results_dir.joinpath("pose_estimation")
video_dir = dlc_dir.joinpath("Videos", "p14_isolation_cropped")
locomotion_dir = results_dir.joinpath("locomotion")
# print(dlc_dir, scripts_dir, pose_dir, video_dir)

# 2: Create a list of csv files in the pose estimation directory, the names of those subjects, and their videos
csvs = lf.csv_list(pose_dir)
## testing: only use first individual in list
# csvs = csvs[0:1]
subject_list_path = results_dir.joinpath("subject_list.csv")
subjects = pd.read_csv(subject_list_path, header=None)
videos = [video_dir.joinpath("p14_isolation_"+name+".mp4") for name in subjects]

# 3: Label videos and write info to a .pkl file
for i, ind in enumerate(subjects):
    file_path = Path(f"{dlc_dir}/Results/p14_oxtrko/locomotion/roi_coordinates/{ind}_cup_coords.pkl")
    if file_path.is_file():
        print(f"{ind}'s video has already been labeled.")
    else:
        x1, y1, x2, y2, pix_per_cm_x, pix_per_cm_y = lf.select_roi(videos[i], "Draw rectangle to outline base of cup.")
        data = {"pix_per_cm_x": pix_per_cm_x, "pix_per_cm_y": pix_per_cm_y, "x1": x1, "x2": x2, "y1": y1, "y2": y2}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully pickled and saved to {file_path}")