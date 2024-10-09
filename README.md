# OLS_analysis
Single-molecule tracking analysis code for use with oblique line-scan (OLS) microscope

## declutter.py
Script that declutters long TIF file names and moves all non-TIF files to an "other" folder

## quot_fast_track.py
Script from Vinson Fan for fast tracking of files with quot.

## settings.toml
Settings file for quot particle tracking

## run_saspt_byprefix.py
Runs SASPT on all TIF files with the same prefix. The prefix is defined as the part of the file name before the first digit, so you will need to modify this if your prefix contains numbers!

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
