**Locomotion Analysis**

This is my recommended workflow for conducting a basic locomotion analysis of DeepLabCut pose estimation data: 

1) Label your regions of interest using *label_rois.py*. 
2) Calculate distance traveled each second using *distance_traveled.py*. This will be your base (unless you are interested in < 1 sec time periods). 
3) Conduct further analyses: e.g. binning distance traveled by larger amounts, calculating velocity, plotting, and statistical analyses. Do these in separate files (*not yet created*). 