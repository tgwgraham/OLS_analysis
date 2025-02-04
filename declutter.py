from glob import glob
import os
from shutil import move
import re
	


def declutter(directory,nfields,prefixes):

    os.makedirs(directory + '/other',exist_ok=True)

    # write current list of files to a text file
    # List all files in the directory
    files = [f for f in os.listdir(directory + '/')] 

    # Write the list of files to the output text file
    with open(directory + '/other/oldfilelist.txt', 'w') as file:
        for filename in files:
            file.write(filename + '\n')


    # move everything that isn't a TIF file or python script to other
    for f in files:
        file_path = os.path.join(directory, f)
        if os.path.isfile(file_path) and f[-4:] != '.tif' and f[-3:] != '.py':
            move(directory + "/" + f,directory + '/other/')


    # Regex pattern to match the filenames and extract parts
    regex_string = r'(.*)Iter_' + r'(\d{4})_'*nfields + r'.*\.tif'

    pattern = re.compile(regex_string)

    # Iterate through all files in the directory
    for filename in os.listdir(directory):

        match = pattern.match(filename)
        if match:
            # Extract the relevant parts from the filename
            new_filename = match.group(1)  # The part before "Iter_"
            
            for j in range(nfields):
                new_filename = new_filename + match.group(j+2)
                if j<nfields-1:
                    new_filename = new_filename + "_"

            new_filename = new_filename + ".tif"
            
            # Rename the file
            os.rename(directory + "/" + filename, directory + "/" + new_filename)

        
    for prefix in prefixes:
        os.makedirs(f'{directory}/{prefix}',exist_ok=True)
        os.system(f'mv {directory}/{prefix}*tif {directory}/{prefix}/')


if __name__ == '__main__':
    nfields = 1 # be very careful not to set this too low!
    prefixes = ['v','g','S','H'] # file name prefixes of tif files
    
    # enter one folder per line
    # It is convenient to use the "ls -1d" command at the command line to list directories and then copy and paste here.
    folders = """run1
    run2
    run3"""
    
    for folder in folders.split():
        declutter(f'{folder}/',nfields,prefixes)

