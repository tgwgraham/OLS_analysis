# OLS_analysis
Single-molecule tracking analysis code for use with oblique line-scan (OLS) microscope

# Basic workflow
1. Declutter files using declutter2.py - typically run this from the folder containing all of the runs for that experiment.
2. Create an "analysis" folder within each run, and copy into it the following scripts:
- quot_fast_track2.py
- segmentation2/ols_seg2.py
- segmentation2/sort_all.py
- run_saspt_byprefix.py
2. Track movies using quot_fast_track2.py (typically run this from an analysis folder within the folder for each run).
3. Segment snapshot images using ols_seg2.py (typically run this from an analysis folder within the folder for each run).
4. Sort trajectories by cell using sort_all.py
5. Run SASPT on sorted trajectories using run_saspt_byprefix.py.

# Contents

## declutter.py
Script that declutters long TIF file names, moves them to subfolders by prefix, and moves all non-TIF files to an "other" folder

## quot_fast_track.py
Modification of script from Vinson Fan for fast tracking of files with quot.

## settings.toml
Settings file for quot particle tracking

## run_saspt_byprefix.py
Runs SASPT on all CSV files with the same prefix. The prefix is defined as the part of the file name before the first digit, so you will need to modify this if your prefix contains numbers!

## run_saspt_files.py
Wrapper script to run SASPT on a specific set of files. 

## track_and_saspt.sh
Short bash script to run tracking and SASPT.

## segmentation
### ols_seg.py
Script that segments all TIF files in a folder.

### run_ols_seg.sh
Wrapper script that runs ols_seg.py

### segmentation_settings.toml
Settings file for segmentation.

### sort_all.py
Sorts trajectories from tracked CSV files by cell. Makes a folder for each field of view containing one CSV file per cell. 0.csv contains background trajectories not in any cell.

### check_sorting.ipynb
Can be used to make overlay plots to check the quality of trajectory sorting.
