import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from cellpose import models
from cellpose.io import imread
import skimage
from PIL import Image
from skimage.measure import label, regionprops
from skimage.io import imread
from scipy.ndimage import sum as ndi_sum
import matplotlib.colors as mcolors
import cv2
from glob import glob



# TG 20241125 - new version of segmentation code that uses cellpose
# 1) input folder name
# 2) output folder name

def getperimeterfraction(labels,width=10):
    # Get fraction of the boundary of each mask in array labels within width of the boundary of the image
    # Useful for throwing away anything that is cut off
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(labels, mode='inner')
    edges = labels * boundaries
    perimeter_fraction = np.zeros(labels.max())
    frame = np.ones(labels.shape)
    frame[width:-width,width:-width] = 0
    for j in range(1,labels.max()+1):
        perimeter_fraction[j-1] = ((edges==j) & (frame==1)).sum() / (edges==j).sum()
    return(perimeter_fraction)

def getarea(labels):
    # Get the areas of each ROI in the label array
    return(np.array([(labels==j).sum() for j in range(1,labels.max() + 1)]))
    
def segment_and_filter(fname, outf = None, gauss_sigma=10, diameter=80, 
                edgewidth = 10, perimthresh = 0.15, areathresh = 5000,
                other_imfnames = None, pretty_output=False,
                ):
    # fname - file name
    # outf - base file name for output
    # gauss_sigma - standard deviation of gaussian to use for gaussian blurring
    # diameter - diameter to use for cell finding in Cellpose
    # edgewidth - edge width to use for excluding boundary cells
    # perimthresh - max fraction of the cell perimeter that can be within edgewidth of the edge of the image
    # areathresh - minimum cell area
    # other_imfnames - names of other image files for getting their integrated intensities.
    # pretty_output - whether to make pretty overlay image

    # returns a tuple with:
    # 1) image array
    # 2) dataframe with image properties

    # If outfname != None, then it saves the segmented, filtered image array to 
    
    img = imread(fname)
    img_for_analysis = skimage.filters.gaussian(img, sigma=gauss_sigma)
    model = models.Cellpose(model_type='cyto') # Choosing the pre-trained "cyto" Cellpose model
    channels = [[0,0]]
    labels, flows, styles, diams = model.eval(img_for_analysis, diameter=80, channels=channels)

    pf = getperimeterfraction(labels,width=edgewidth)
    areas = getarea(labels)

    sel = np.isin(labels,np.where(pf<perimthresh)[0]+1)
    sel = sel & np.isin(labels,np.where(areas > areathresh)[0]+1)
    #rej = np.isin(labels,np.where(pf>=perimthresh)[0]+1)
    #rej = rej & np.isin(labels,np.where(areas <= areathresh)[0]+1)
    
    sel_labels = labels*sel
    
    

    
    # Extract properties and store them in a DataFrame
    # Calculate properties of labeled regions
    properties = regionprops(sel_labels)
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
    integrated_intensity = ndi_sum(img, labels=sel_labels, index=region_df['label'])

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

    if outf is not None: # write output
        output_image = Image.fromarray(sel_labels.astype('uint16'))
        output_image.save(f'{outf}.tif')
        
        # write out region properties
        region_df.to_csv(f'{outf}.csv')
    
        if pretty_output: # make easy to read overlays of segmented ROIs on the original image
            plot_labeled_contours(img, sel_labels, region_df, outfname=f'{outf}_overlay.jpg')
            plt.close('all')

    return (labels*sel, region_df)
    
    

def plot_labeled_contours(image, label_array, regiondf, outfname=None):
    # Convert the image to grayscale for plotting if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Plot the image using imshow
    plt.figure(figsize=(20,20))
    plt.imshow(gray_image, cmap='gray')
    
    # Get the unique labels (excluding background 0)
    unique_labels = np.unique(label_array)
    unique_labels = unique_labels[unique_labels != 0]

    # Generate a color map for different labels
    colormap = plt.cm.get_cmap('hsv', len(unique_labels))
    norm = mcolors.Normalize(vmin=min(unique_labels), vmax=max(unique_labels))

    for label in unique_labels:
        # Create a binary mask for the current label
        binary_mask = (label_array == label).astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the color for the current label
        color = colormap(norm(label))[:3]  # Get RGB values from the colormap

        # Plot each contour
        for contour in contours:
            contour = contour.squeeze()
            if contour.ndim == 2:  # Ensure the contour has correct shape
                contour = np.vstack([contour, contour[0]])
                plt.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1)

                # Overlay the text for each region
    for _, row in regiondf.iterrows():
        # Extract information from the DataFrame
        label_value = int(row['label'])
        x_centroid = row['x_centroid']
        y_centroid = row['y_centroid']
        #fraction_within_bounds = row['fraction_within_bounds']

        # Overlay the label (in white)
        plt.text(y_centroid,x_centroid, str(label_value),
                 color='white', fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
    # Show the plot with the overlaid contours
    plt.axis('off')
    
    if outfname is not None:
        plt.savefig(outfname,dpi=300,bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    
    basefname = '../H'
    outfolder = 'masks'
    
    os.makedirs(outfolder,exist_ok=True)
    
    fnames = glob(basefname + "*tif")
    
    for fname in fnames:    
        print(f"Analyzing {fname}")
        outf = f"{outfolder}/{os.path.split(fname)[-1][:-4]}"
        segment_and_filter(fname, outf = outf, gauss_sigma=10, diameter=80, 
                    edgewidth = 20, perimthresh = 0.1, areathresh = 5000,
                    other_imfnames = None, pretty_output=True,
                    )
    
    
    # infolder = sys.argv[1]
    # outfolder = sys.argv[2]
    
    # fnames = glob(infolder + '*tif')
    # os.makedirs(outfolder,exist_ok=True)

    #Loop over all TIF files in the input folder
    # for f in fnames:
        # segmentfile(
            # fname=f,
            # outf=f'{outfolder}/{os.path.basename(f)[:-4]}',
            # thresh1=config.get('thresh1', 120.0),
            # thresh2=config.get('thresh2', -5),
            # rad1=config.get('rad1', 3),
            # rad2=config.get('rad2', 11),
            # size_threshold=config.get('size_threshold', None),
            # ybounds=config.get('ybounds', None),
            # other_imfnames=config.get('other_imfnames', None),
            # dilation_strelsize=config.get('dilation_strelsize', 3),
            # dilation_nrounds=config.get('dilation_nrounds', 3),
            # pretty_output=config.get('pretty_output', True)
        # )
   