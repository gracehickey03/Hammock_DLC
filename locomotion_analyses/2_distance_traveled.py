# Created: Jan 10, 2026
# Last modified: Feb 1, 2026
# Last git commit: Feb 1, 2026
## Updates: Switched to per-frame distances, binned to true seconds

import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import locomotion_helper_functions as lf
import cv2

# -------------------------------------------------
# 1: Directories
# -------------------------------------------------
dlc_dir = Path(__file__).parent.parent.resolve()
results_dir = dlc_dir.joinpath("Results", "p14_oxtrko")
pose_dir = results_dir.joinpath("pose_estimation")
video_dir = dlc_dir.joinpath("Videos", "p14_isolation_cropped")
locomotion_dir = results_dir.joinpath("locomotion")

print(f"DLC Dir: {dlc_dir}")
print(f"Pose Estimation Dir: {pose_dir}")
print(f"Video Dir: {video_dir}")

# -------------------------------------------------
# 2: Subjects and files
# -------------------------------------------------
# csvs = lf.csv_list(pose_dir)

subject_list_path = results_dir.joinpath("subject_list.csv")
subjects = pd.read_csv(subject_list_path, low_memory=False)["subject"].tolist()

bodyparts = [
    "nose", "head_midpoint", "mouse_center", "tail_base", "tail3",
    "tail_end", "left_shoulder", "right_shoulder",
    "left_midside", "right_midside", "left_hip", "right_hip"
]

# subjects = [""] # shortened list for testing 
# -------------------------------------------------
# 3: Per-frame → per-second locomotion
# -------------------------------------------------
for i, ind in enumerate(subjects):

    export_path = locomotion_dir / "second_movement" / f"{ind}_sec_dists.csv"
    if export_path.is_file():
        print(f"{ind}: second-by-second file already exists.")
        continue

    cup_coord_filename = (
        locomotion_dir / "roi_coordinates" / f"{ind}_cup_coords.pkl"
    )
    if not cup_coord_filename.is_file():
        print(f"{ind}: ROI not labeled. Skipping.")
        continue
    with open(cup_coord_filename, "rb") as f:
        roi_coord_data = pickle.load(f)

    # pose estimation path 
    pose_key = str(ind) + "DLC"
    matches = list(pose_dir.glob(f"*{pose_key}*.csv"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No CSV found containing '{ind}'")
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple CSVs found containing '{ind}': {matches}")

    csv_path = matches[0]


    pix_per_cm_x = roi_coord_data["pix_per_cm_x"]
    pix_per_cm_y = roi_coord_data["pix_per_cm_y"]
    fps = roi_coord_data["fps"]

    dfs_sec = []

    for part in bodyparts:
        # -----------------------------------------
        # Load bodypart coordinates
        # -----------------------------------------
        x, y, p = lf.import_bodypart(csv_path, part)
        # video frame count
        video = cv2.VideoCapture(video_dir / f"p14_isolation_{ind}.mp4")
        n_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()


        # truncate to correct video length
        x = x[:n_video_frames]
        y = y[:n_video_frames]  
        p = p[:n_video_frames]  

        # csv frame count
        n_csv_frames = len(x)


        # -----------------------------------------
        # Per-frame distances (cm per frame)
        # -----------------------------------------
        dx = np.diff(x) / pix_per_cm_x
        dy = np.diff(y) / pix_per_cm_y
        frame_dists = np.sqrt(dx**2 + dy**2)

        # -----------------------------------------
        # Map frames to real seconds
        # -----------------------------------------
        frame_times = np.arange(len(frame_dists)) / fps
        sec_idx = np.floor(frame_times).astype(int)

        sec_dists = np.bincount(sec_idx, weights=frame_dists)

        n_sec = len(sec_dists)

        df_part = pd.DataFrame(
            {"distance": sec_dists},
            index=pd.MultiIndex.from_arrays(
                [
                    np.repeat(ind, n_sec),
                    np.repeat(part, n_sec),
                    np.arange(n_sec)
                ],
                names=["name", "bodypart", "second"]
            )
        )

        dfs_sec.append(df_part)

    # -----------------------------------------
    # Save per-second distances
    # -----------------------------------------
    dfs_sec = pd.concat(dfs_sec)
    dfs_sec.to_csv(export_path)

    print(f"{ind}: saved {n_sec/60:.2f} minutes of second-by-second movement.")
    # print(f"  len(x): {len(x)}")
    # print(f"  expected frames: {n_video_frames}")
    # print(f"  implied minutes: {len(x) / fps / 60:.2f}")

    print(f"  video frames: {n_video_frames}")
    print(f"  csv frames:   {n_csv_frames}")
    # print(f"  video minutes: {n_video_frames / fps / 60:.2f}")
    # print(f"  csv minutes:   {n_csv_frames / fps / 60:.2f}")
