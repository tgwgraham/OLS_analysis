import os, numpy as np
from glob import glob
from skimage import io

def trajbyroi(masktif,trajfile,outfolder,cropbounds=None,write_background=False):
    """
    trajbyroi(masktif,trajfile,outfolder)

    Classifies all of the trajectories in a csv file for each ROI mask in 
    the specified TIF mask file.
    Note that this version can also write out background trajectories as a "0" file, 
    which can be useful for analyzing out-of-mask background.

    Inputs:
        masktif - .tif file that contains the segmented mask
        trajfile - CSV file containing trajectories
        outfolder - output folder for writing out classified csvs
                    It will create a subfolder with the same name as the masktif
        cropbounds - bounds for cropping in the y axis if the SMT FOV was smaller than that used to collect the 
                    image for the mask (default: None)
        write_background - whether or not to write out background region to a "0.csv" file (default: False)

    """    
    
    # column number in which trajectory index is stored in tracking csv files
    TRAJECTORY_COLUMN = 17 
    
    os.makedirs(outfolder,exist_ok = True)
    
    # cell masks in current FOV
    currmask = io.imread(masktif)
    currmask = currmask.astype('double')
    
    if cropbounds is not None:
        currmask = currmask[cropbounds[0]:cropbounds[1],:]
    
    # Loop over the file a first time to get maximum trajectory number in the file
    maxtraj = 0

    with open(trajfile) as fh:
        fh.readline()
        line = fh.readline()
        while line:
            maxtraj = max(int(line.split(',')[TRAJECTORY_COLUMN]),maxtraj)
            line = fh.readline()

    # initialize an array of -1's with that many elements to contain the cell number 
    # for each trajectory (or NaN if the trajectory passes over multiple cells)
    trajcell = -np.ones(maxtraj+1) # array starting at 0 and ending at maxtraj, inclusive

    # loop over the csv file a second time, and determine in which cell mask each trajectory falls
    with open(trajfile) as fh:
        fh.readline()
        line = fh.readline()
        allcelln = set()

        while line:
            linespl = line.split(',')

            # current trajectory number
            trajn = int(linespl[TRAJECTORY_COLUMN])

            # current x and y coordinates
            x = round(float(linespl[1]))
            y = round(float(linespl[0]))

            # get cell number
            # celln = 0 corresponds to background regions. Numbering of cells starts at 1
            celln = currmask[y,x] 

            # add this cell index to the list of all cell indices
            allcelln.add(celln)

            # if it has not yet been classified, classify it to the cell it is in
            if trajcell[trajn] == -1: 
                trajcell[trajn] = celln
            # if it has previously been classified to another cell, set it to nan
            elif trajcell[trajn] != celln:
                trajcell[trajn] = np.nan
            line = fh.readline()

    # loop over the file one last time and write out each line to a file for that cell
    with open(trajfile) as fh:
        header = fh.readline()

        # Create a distinct output folder for this movie
        movie_folder = os.path.basename(trajfile)[:-4]
        os.makedirs(f"{outfolder}/{movie_folder}",exist_ok=True)

        # open output file handles and initialize each with a header row
        fh2 = {}
        for n in allcelln:
            # Open one file handle per cell
            # exclude trajectories in the background region (n = 0)
            if (n>0) or write_background:
                os.makedirs(f"{outfolder}/{movie_folder}",exist_ok=True)
                fh2[n] = open(f"{outfolder}/{movie_folder}/{int(n)}.csv",'w')
                fh2[n].write(header)

        line = fh.readline()
        while line:
            linespl = line.split(',')

            # trajectory number of current localization
            trajn = int(linespl[TRAJECTORY_COLUMN])

            # cell number of current trajectory
            celln = trajcell[trajn]

            # only write out the current localization if it is part of a 
            # trajectory within a cell that is selected
            if not np.isnan(celln) and (celln != 0 or write_background):
                celln = int(celln)
                fh2[celln].write(line)

            line = fh.readline()

        # close all file handles
        for f in fh2.keys():
            fh2[f].close()

if __name__ == '__main__':

    maskprefix = 'H'            # prefix for mask file names
    maskfolder = 'masks'        # folder containing mask TIFs
    trackfolder = '../../tracking/'     # folder containing tracking CSVs
    outfolder = 'sorted_trajectories'   # output folder for CSVs sorted by cell
    prefixes = ['v','d']                # prefixes for SMT files
                                        # My current convention is to name the mask channel "H" followed
                                        # by a number, and name the corresponding SMT file(s) some other
                                        # letter followed by the same number.
    write_background=True       # Whether to write out a "0.csv" file for trajectories outside of any mask    
        
    # get mask TIF file names
    masknames = glob(f'{maskfolder}/{maskprefix}*tif')
    
    # loop over each mask file and sort trajectories from corresponding trajectory file(s)
    for maskname in masknames:
        index = os.path.basename(maskname)[1:-4]
        for prefix in prefixes:
            trajfile = f'{trackfolder}{prefix}{index}_trajs.csv'
            if os.path.exists(trajfile):
                trajbyroi(maskname,trajfile,outfolder,write_background=write_background)
            else:
                print(f'Warning: {trajfile} not found.')
            print(maskname,trajfile)

