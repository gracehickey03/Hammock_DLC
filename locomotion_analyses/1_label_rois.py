# Created: Jan 10 2026
# Last modified: Jan 10 2026
# Last git commit: Jan 10 2026
## Updates: created.

# Use this file to label the region of interest (ROI) for your experiment. 
# Jan 2026: For p14 sensory oxtrko isolation: highlighting the bottom of the cup. 

import cv2
import os
from pathlib import Path
import pickle

## Helper functions
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

    fps = cap.get(cv2.CAP_PROP_FPS)

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
    x1, y1, width, height = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)

    # convert pixels to cm 
    pix_per_cm_x = width/x_diam
    pix_per_cm_y = height/y_diam
    
    return x1, y1, pix_per_cm_x, pix_per_cm_y, fps


# 1: Directories
dlc_dir = Path(__file__).parent.parent.resolve()
scripts_dir = dlc_dir.joinpath("Scripts")
results_dir = dlc_dir.joinpath("Results", "p14_oxtrko")
pose_dir = results_dir.joinpath("pose_estimation")
video_dir = dlc_dir.joinpath("Videos", "p14_isolation_cropped")
locomotion_dir = results_dir.joinpath("locomotion")
print(dlc_dir)
print(scripts_dir) 
print(pose_dir)
print(video_dir)

# 2: Create a list of csv files in the pose estimation directory, the names of those subjects, and their videos
csvs = csv_list(pose_dir)
## testing: only use first individual in list
# csvs = csvs[0:1]

names = [csv.name.split("p14_isolation_")[1].split("DLC")[0] for csv in csvs] 
# check that this line works with file paths in HPG. and there is likely a better way to do it than this. 

videos = [video_dir.joinpath("p14_isolation_"+name+".mp4") for name in names]

# 3: Label videos and write info to a .pkl file
for i, ind in enumerate(names):
    file_path = Path(f"{dlc_dir}/Results/p14_oxtrko/locomotion/roi_coordinates/{ind}_cup_coords.pkl")
    if file_path.is_file():
        print(f"{ind}'s video has already been labeled.")
    else:
        x1, y1, pix_per_cm_x, pix_per_cm_y, fps = select_roi(videos[i], "Draw rectangle to outline base of cup.")
        data = {"pix_per_cm_x": pix_per_cm_x, "pix_per_cm_y": pix_per_cm_y, "x1": x1, "y1": y1, "fps": fps}
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data successfully pickled and saved to {file_path}")