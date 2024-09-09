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
