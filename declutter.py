from glob import glob
import os
from shutil import move
import re
	


def declutter(directory,nfields):

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
	
	
if __name__ == '__main__':
    # This example declutters files with 3 numerical fields in the current directory
	nfields = 3
    directory = "./"
    declutter(directory,nfields)
    
    ## Here's an example of how to declutter files in multiple directories called "run1" through "run6"
    ## with three numeric fields
    # nfields = 3
	# for directory in ["run" + str(j) for j in range(1,7)]:
	#	declutter(directory,nfields)


