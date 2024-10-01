import cv2
from skimage import io
import numpy as np
from scipy.ndimage import binary_fill_holes
from PIL import Image
from scipy.ndimage import label, find_objects, sum as ndi_sum
from skimage.measure import label, regionprops
import pandas as pd
import sys
import toml
import os
from glob import glob

# TG 20241001 - modified the main script to process all the TIF files in a folder
# new input command line arguments are:
# 1) input folder name
# 2) output folder name
# 3) settings file name

def dilate_labels(label_array,strelsize=3,nrounds=3):
    # dilates regions labeled with different numbers in a 2D label_array
    # Dilation only occurs into pixels with a value of 0, so that labeled regions do not
    # invade other labeled regions.
    #
    # strelsize - size of structuring element to use for dilation
    # nrounds - number of times to apply the dilation operation
    
    
    # Get the unique labels (ignoring the background label 0)
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude the background label (0)

    # Create an empty array to store the dilated labels
    dilated_labels = np.copy(label_array)

    # Define a 3x3 structuring element
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (strelsize,strelsize))

    for j in range(nrounds):
        for label in unique_labels:
            # Create a binary mask for the current label
            binary_mask = (label_array == label).astype(np.uint8)
        
            # Dilate the binary mask
            dilated_mask = cv2.dilate(binary_mask, structuring_element)
            
            # Update the dilated_labels array: only dilate into background areas
            dilated_labels[(dilated_mask == 1) & (label_array == 0)] = label
            
            label_array = dilated_labels
            
    for label in unique_labels:
        binary_mask = (label_array == label).astype(np.uint8)
        binary_mask = binary_fill_holes(binary_mask)
        label_array[(binary_mask == 1) & (label_array == 0)] = label

    return label_array

def segmentfile(fname,
                thresh1=120.0, thresh2=-5, rad1=3, rad2=11,
                outf=None,size_threshold=None,
                miny = None, maxy = None,
                other_imfnames = None,
                dilation_strelsize=3,dilation_nrounds=3,
               ):
    # segments image to produce nuclear ROIs for further analysis
    # thresh1 - minimum intensity threshold for nuclei
    # thresh2 - maximum intensity threshold for excluding nuclear boundaries
    # rad1 - radius for narrow Gaussian blur
    # rad2 - radius for wide Gaussian blur
    # outf - optional; base file name for output TIF and CSV files. No output if this is set to None.
    # size_threshold - minimum size for ROIs.
    # miny and maxy - minimum and maximum bounds (min inclusive, max not) for smaller ROI.
    #                 This is useful if you use an imaging ROI that is smaller than the pre-image
    # other_imfnames - list of other image file names for getting integrated intensities within ROIs in
    #                 those images (e.g., other fluorescence channels)
    # dilation_strelsize - size of structuring element to use for dilation    
    # dilation_nrounds - number of times to apply the dilation operation
    
    # Convert thresholds to appropriate types
    thresh1 = float(thresh1)
    thresh2 = float(thresh2)
    rad1 = int(rad1)
    rad2 = int(rad2)
    
    image = io.imread(fname)
    image = image.astype('double')
    
    # generate inverted image for reciprocal mask
    immax = image.max()
    invim = immax-image

    # Apply Gaussian blur with two different radii
    blurred_image = cv2.GaussianBlur(image, (21,21), rad1)
    blurred_image2 = cv2.GaussianBlur(image, (21,21), rad2)
    
    filt1 = blurred_image-blurred_image2
    filt3 = (filt1 < thresh2)
    _, filt2 = cv2.threshold(blurred_image, thresh1, 1, cv2.THRESH_BINARY)
    filled_mask = binary_fill_holes(filt2-filt3 > 0)
    
    num_labels, labeled_image = cv2.connectedComponents(filled_mask.astype(np.uint8))

    dilabel = dilate_labels(labeled_image)
    
    if size_threshold is not None:
        # Find the size of each region
        region_sizes = ndi_sum(np.ones_like(dilabel), dilabel, range(1, dilabel.max() + 1))

        # Identify the labels of the regions that meet the size threshold
        large_region_labels = np.where(region_sizes >= size_threshold)[0] + 1

        # Create a mask to keep only the large regions
        large_regions_mask = np.isin(dilabel, large_region_labels)

        # Apply the mask to keep only large regions
        dilabel_filtered = dilabel * large_regions_mask

        # Manually relabel the remaining regions sequentially
        unique_labels = np.unique(dilabel_filtered)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        # Create a mapping from old labels to new sequential labels
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        # Apply the mapping to relabel the array
        dilabel_relabelled = np.copy(dilabel_filtered)
        for old_label, new_label in label_mapping.items():
            dilabel_relabelled[dilabel_filtered == old_label] = new_label
            
        dilabel = dilabel_relabelled
        
    # Extract properties and store them in a DataFrame
    # Calculate properties of labeled regions
    properties = regionprops(dilabel)
    region_data = {
        "label": [],
        "area": [],
        "centroid": [],
        "perimeter": []
    }

    for prop in properties:
        region_data["label"].append(prop.label)
        region_data["area"].append(prop.area)
        region_data["centroid"].append(prop.centroid)
        region_data["perimeter"].append(prop.perimeter)

    # Convert the dictionary to a pandas DataFrame
    region_df = pd.DataFrame(region_data)
    
    # Split the centroid column into x_centroid and y_centroid
    region_df[['x_centroid', 'y_centroid']] = pd.DataFrame(region_df['centroid'].tolist(), index=region_df.index)

    # Drop the original centroid column if you no longer need it
    region_df = region_df.drop(columns=['centroid'])
    
    # Calculate the integrated intensity for each labeled region
    integrated_intensity = ndi_sum(image, labels=dilabel, index=region_df['label'])

    # Add the integrated intensity to the DataFrame
    region_df['integrated_intensity'] = integrated_intensity
    
    # iterate over other image file names, if any are provided, and get integrated intensities in
    # each of those channels; append to the dataframe
    if other_imfnames is not None:
        for j,imfname in enumerate(other_imfnames):
            currim = io.imread(imfname)
            currim = currim.astype('double') 
            integrated_intensity = ndi_sum(currim, labels=dilabel, index=region_df['label'])
            region_df[f'integrated_intensity_{j+1}'] = integrated_intensity
            
    # if y-bounds for smaller imaging region are specified, then get the fraction of pixels for each ROI that
    # fall within that smaller imaging region
    
    if miny is not None and maxy is not None:
        # Get the coordinates of all pixels in the labeled array
        coords = np.array(np.nonzero(dilabel)).T  # This gives array of [y, x] pairs

        # Create a dictionary to store the fraction of pixels within the bounds for each label
        fraction_within_bounds = {}

        for label_value in region_df['label']:
            # Get the y-coordinates of pixels with the current label
            y_coords = coords[dilabel[coords[:, 0], coords[:, 1]] == label_value][:, 0]

            # Count how many of those y-coordinates are within the bounds
            within_bounds_count = np.sum((y_coords >= miny) & (y_coords <= maxy))

            # Total count of pixels with the current label
            total_count = len(y_coords)

            # Calculate the fraction within bounds
            fraction_within_bounds[label_value] = within_bounds_count / total_count if total_count > 0 else 0

        # Add the fraction to the DataFrame
        region_df['fraction_within_bounds'] = region_df['label'].map(fraction_within_bounds)
    
        
    if outf is not None: # write output
        image = Image.fromarray(dilabel.astype('uint16'))
        image.save(f'{outf}.tif')
        
        # write out region properties
        region_df.to_csv(f'{outf}.csv')
    
    return dilabel, region_df 


if __name__ == '__main__':

    config = toml.load(sys.argv[3])
    
    infolder = sys.argv[1]
    outfolder = sys.argv[2]
    
    fnames = glob(infolder + '/*tif')
    os.makedirs(outfolder,exist_ok=True)

    # Loop over all TIF files in the input folder
    for f in fnames:
        segmentfile(
            fname=f,
            outf=f'{outfolder}/{os.path.basename(f)[:-4]}',
            thresh1=config.get('thresh1', 120.0),
            thresh2=config.get('thresh2', -5),
            rad1=config.get('rad1', 3),
            rad2=config.get('rad2', 11),
            size_threshold=config.get('size_threshold', None),
            miny=config.get('miny', None),
            maxy=config.get('maxy', None),
            other_imfnames=config.get('other_imfnames', None),
            dilation_strelsize=config.get('dilation_strelsize', 3),
            dilation_nrounds=config.get('dilation_nrounds', 3)
        )
        