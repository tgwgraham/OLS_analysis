from glob import glob


sorted_folder = 'sorted_trajectories'

def rewrite_trajectories(input_files, output_file, start_frame = None, end_frame = None):

    # pool all trajectories from a list of trajectory files and write them out to a single file
    # Writes out only x, y, frame, and trajectory columns, and renumbers trajectories so that their numbers are all unique
    #
    # If start_frame and/or end_frame are specified, then restricts the frame numbers that are included
    # start_frame and end_frame are zero-indexed, and both are inclusive.
 
    maxtraj = 0
    
    with open(output_file, 'w') as outfile:

        outfile.write('x,y,frame,trajectory\n')
        
        for input_file in input_files:
            # Open the input and output files
            with open(input_file, 'r') as infile: 
            
                currmaxtraj = 0
                
                # Skip the header row
                next(infile)

                # Process each line
                for line in infile:
                    # Split the line into fields
                    fields = line.strip().split(',')
                    
                    # Extract fields 0, 1, 15, and 17
                    #extracted_fields = [fields[i] for i in (0, 1, 15, 17)]
                    
                    x = float(fields[0])
                    y = float(fields[1])
                    frame = int(fields[15])
                    trajectory = int(fields[17])
                    
                    if ((start_frame is None) or (frame >= start_frame)) and ((end_frame is None) or (frame <= end_frame)):
                        
                        if trajectory > currmaxtraj:
                            currmaxtraj = trajectory
                        
                        outfile.write(f"{x},{y},{frame},{trajectory+maxtraj}\n")
            
            maxtraj = maxtraj + currmaxtraj + 1
            


if __name__ == "__main__":
    
    sorted_folder = 'sorted_trajectories'
    
    maxfnum = None # set this to something other than None if you want to impose a limit on the maximum file number (e.g., for including only time points prior to drug treatment)

    prefixes = {'g':{'start_frame':40,'end_frame':70},
                'v':{'start_frame':31,'end_frame':61}
                }
    
    
    for prefix in prefixes.keys():    
        if maxfnum is not None:
            traj = []
            for j in range(maxfnum+1):
                traj.extend(glob(f"{sorted_folder}/{prefix}/{prefix}_{j:04d}*/*csv")) 
        else:
            traj = glob(f"{sorted_folder}/{prefix}/{prefix}_*/*csv")
        rewrite_trajectories(traj, f'{sorted_folder}/all_{prefix}.csv', 
                                    start_frame=prefixes[prefix]['start_frame'], 
                                    end_frame=prefixes[prefix]['end_frame'])

