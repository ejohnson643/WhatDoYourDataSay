"""
################################################################################
##	ES_APPM 375 Assignment 2 Problem 4 Image Processing Package
################################################################################
  
	Author: Eric Johnson
	Date Created: Saturday, January 25, 2020
	Email: ericjohnson1.2020@u.northwestern.edu

################################################################################
 
	Overview:
		This package contains functions that will be useful in extracting the
		YFP fluorescence from images of E. coli using simultaneous mCherry
		fluorescence or phase contrast images.  

		This will be useful in replicating some of the work done in the paper by
		Garcia and Phillips, "Quantiative dissection of the simple repression 
		input-output function" from 2011.  The goal of this paper is to use a 
		thermodynamic model to  estimate in vivo repressor number or repressor 
		binding energy.  They make these estimates using fluorescence images of
		E. coli in different mutated strains.  In order to generate the fold-
		change data needed to make these estimates, one needs to first do some 
		processing of the fluorescence images.  This package provides a few
		functions that make that processing happen.

	Functions:
		strainFileFinder
		channelPlotter
		channelPixelDistPlotter
		getYFPofCells
		calcFoldChange

################################################################################
################################################################################
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
import seaborn as sns
import skimage.measure as skm
import skimage.morphology as morph
import skimage.filters as skf

sns.set(color_codes=True)
sns.set_style({'legend.frameon':True, 'legend.facecolor':'w'})
matplotlib.rc("font", size=20)
matplotlib.rc("xtick", labelsize=20)
matplotlib.rc("ytick", labelsize=20)
matplotlib.rc("axes", labelsize=24)
matplotlib.rc("axes", titlesize=28)
matplotlib.rc("legend", fontsize=14)
matplotlib.rc("figure", titlesize=24)

strainNames = ['Auto', 'Delta', 'HG104', 'RBS1', 'RBS1027', 'RBS1147', 'RBS446']
channelNames = ['YFP', 'mCherry', 'Phase']

# These are from Garcia & Phillips (2011)
ReprCopyNo = {"HG104":11,
		  "RBS1147":30,
		  "RBS446":62,
		  "RBS1027":130,
		  "RBS1":610}
ReprCopyNoErr = {"HG104":2,
		  "RBS1147":8,
		  "RBS446":20,
		  "RBS1027":40,
		  "RBS1":70}

dataDir = "Phillips_Garcia_Data/laci_full_set"

def strainFileFinder(strain, dataDir=dataDir):
	"""strainFileFinder(strain, dataDir="Phillips_Garcia_Data/laci_full_set")

	Finds phase-contrast, YFP fluorescence, and mCherry fluorescence .tif files.

	Inputs:
	=======
		strain 		(str) Name of strain whose files you want to load
		dataDir		(str) Path to location of strain folders.  Optional; default
					is "Phillips_Garcia_Data/laci_full_set".

	Outputs:
	========
		YFPImgs 	(list of str) List of paths to YFP fluorescence images of
					requested E. coli strain.
		mChrImgs 	(list of str) List of paths to mCherry fluorescence images 
					of requested E. coli strain.
		PhaseImgs 	(list of str) List of paths to phase-contrast images of
					requested E. coli strain.
	"""

	## Check inputs.
	if not isinstance(strain, str):
		raise TypeError("Input 'strain' should be a string!")
	if strain not in strainNames:
		raise ValueError(f"Unrecognized strain = {strain}.  Accepted strains: "+
			", ".join(strainNames))
	if not isinstance(dataDir, str):
		raise TypeError("Input keyword 'dataDir' should be a string!")
	if not os.path.isdir(dataDir):
		raise FileError(f"dataDir = {dataDir} is not a valid path/directory!")

	## Load all files in dataDir/strain
	files = os.listdir(os.path.join(dataDir, strain))

	## Get all the .tif files.
	TIFfiles = sorted([f for f in files if ".tif" in f])

	## Get the phase, YFP, and mCherry images separately.
	phaseImgs = sorted([os.path.join(dataDir, strain, f)
		for f in TIFfiles if "Phase" in f])
	YFPImgs = sorted([os.path.join(dataDir, strain, f)
		for f in TIFfiles if "YFP" in f])
	mChrImgs = sorted([os.path.join(dataDir, strain, f)
		for f in TIFfiles if "mCherry" in f])

	return YFPImgs, mChrImgs, phaseImgs


def channelPlotter(filePaths, axes=None, fileNo=None):
	"""channelPlotter(filePaths, axes=None, fileNo=None)

	Plots figure showing .tif images.

	Inputs:
	=======
		filePaths	(list of str) List of paths to .tif images to plot.
		axes		(matplotlib.pyplot.axes object) Axes on which to plot
					images.  Optional; if none provided, generates 2-row set of
					subplots for full list of filePaths.
		fileNo		(integer) Number of file in filePaths to plot.  Optional; if
					not provided, plots all images in list.  If provided, and
					axes are not provided, creates a 1x1 figure with image.

	Outputs:
	========
		fig 		(figure object) Figure handle for figure.
		axes 		(axes object(s)) Lists axes handles on which images are
					plotted.
	"""

	## If the user has specified a fileNumber
	if fileNo is not None:
		if fileNo not in range(len(filePaths)):
			raise ValueError("Optional argument 'fileNo' must be int in "+
				f"[0, {len(filePaths)}].")

		## If the user has not provided axes
		if axes is None:
			fig, axes = plt.subplots(1, 1, figsize=(8, 6))
			nRows, nCols = 1, 1

	## If the user has not provided axes
	if axes is None:
		nRows = 2
		nCols = int(np.ceil(float(len(filePaths))/nRows))
		fig, axes = plt.subplots(nRows, nCols, figsize=(16, 7))
	## If they have, figure out how many rows and columns.
	else:
		try:
			nRows = len(axes)
			try:
				nCols = len(axes[0])
			except:
				nCols = 1
		except:
			nRows, nCols = 1, 1

	## Loop through the file paths
	rowNo, colNo = 0, 0
	for fNo, filePath in enumerate(filePaths):

		## If the user has specified a file number, skip until that file number
		if fileNo is not None:
			if fNo != fileNo:
				continue

		## If there are more files than there are subplots, skip them!
		if fNo >= nRows * nCols:
			continue

		## Check that every path is valid and is a .tif image.
		if not os.path.isfile(filePath):
			raise FileError(f"Could not load file {filePath}!")
		if filePath[-4:] != '.tif':
			raise ValueError(f"File {filePath} is not a .tif image!")

		## Get just the file name from the full path.
		if "/" in filePath:
			brokenName = filePath.split("/")[-1]
		else:
			brokenName = filePath.split("\\")[-1]

		## Detect what kind of image it is.
		if "YFP" in brokenName:
			chan = 'YFP'
		elif "mCherry" in brokenName:
			chan = 'mCherry'
		elif "Phase" in brokenName:
			chan = "Phase"
		else:
			raise ValueError(f"File name {brokenName} doesn't have recognized "+
				"channel name... ({", ".join(channelNames)})")

		## Try and get the correct set of axes.
		try:
			ax = axes[rowNo]
			try:
				ax = ax[colNo]
			except:
				pass
		except:
			pass

		## Load in the image
		img = plt.imread(filePath)

		## Certain image types are masked for viewing purposes.  Try commenting
		## this out and see what happens!
		if chan == 'YFP':
			mask = img > 200
			img[mask] = np.max(img[~mask])
		elif chan == 'Phase':
			mask = img > 750
			img[mask] = np.max(img[~mask])

		## Show the image
		ax.imshow(img)

		## Remove the grid and axes ticks.
		ax.grid(False)
		ax.set_xticks([])
		ax.set_yticks([])

		## Set the title.
		ax.set_title(brokenName)

		## Increment the column and row counters.
		colNo += 1
		if colNo >= nCols:
			colNo = 0
			rowNo += 1

	## Get the figure object if we don't have it, then use tight_layout to make
	## things look nicer.
	fig = plt.gcf()
	fig.tight_layout()

	return fig, axes


def channelPixelDistPlotter(imagePath, axes=None):
	"""channelPixelDistPlotter(imagePath, axes=None)

	Plots pixel distribution (normalized PDF and log-counts) of .tif image at
	'imagePath.'


	Inputs:
	=======
		imagePath	(str) Path to .tif image to plot.
		axes		(matplotlib.pyplot.axes object) List of 2 axes on which to 
					plot images.  Optional; if none provided, generates 2 panel 
					figure to show both a PDF (with KDE) and log-counts 
					histogram.

	Outputs:
	========
		fig 		(figure object) Figure handle for figure.
		axes 		(axes object(s)) Lists axes handles on which images are
					plotted.
	"""

	if not os.path.isfile(imagePath):
		raise FileError(f"Could not find file {imagePath}!")
	if imagePath[-4:] != '.tif':
		raise FileError(f"Image should be a .tif file.")

	if axes is None:
		fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
	else:
		assert len(axes) == 2, "Provided 'axes' are expected to be a list of 2."

	img = plt.imread(imagePath)

	h1 = sns.distplot(img.ravel(), ax=axes[0], label='Data',
		kde_kws={"label":"K.D.E."})

	histVals = [h.get_height() for h in h1.patches]

	kdex, kdey = h1.get_lines()[0].get_data()

	vals, bins = np.histogram(img.ravel(), bins=len(histVals))

	kdey = kdey * len(img.ravel()) * np.mean(np.diff(bins)) + 1

	axes[1].bar(bins[:-1], vals, width=np.mean(np.diff(bins)),
		alpha=0.5, align='edge', label='Data')
	axes[1].plot(kdex, kdey, color='b', label='K.D.E.')
	axes[1].set_yscale('log')

	axes[0].set_ylabel("Frequency", fontsize=20)
	axes[1].set_ylabel("No. Observations", fontsize=20)
	axes[1].set_xlabel("Fluorescence Intensity (a.u.)")

	axes[0].legend(fontsize=16, loc=1)
	axes[1].legend(fontsize=16, loc=1)

	[tick.label.set_fontsize(14) for tick in axes[0].xaxis.get_major_ticks()]
	[tick.label.set_fontsize(14) for tick in axes[0].yaxis.get_major_ticks()]
	[tick.label.set_fontsize(14) for tick in axes[1].xaxis.get_major_ticks()]
	[tick.label.set_fontsize(14) for tick in axes[1].yaxis.get_major_ticks()]

	fig = plt.gcf()
	fig.tight_layout()

	return fig, axes


def getYFP_Image(YFP, img, imgtype, threshold_op=None, verbose=0,
                  show_plots=False, remove_outliers=True,
                  minArea=None):
    """Gets the YFP values of cells detected from a corresponding phase-
    contrast or mCherry image.
    
    Specifically, this function will apply various image processing functions
    to the input mCherry or phase-contrast image to detect cells.  These
    functions include thresholding, morphological filtering, and size-
    filtering.  Once the cells have been detected, the corresponding pixels in
    the YFP image are selected, and their fluorescences are calculated.
    
    Parameters
    ----------
        YFP: 2D numpy array
            YFP fluorescence image.
        img: 2D numpy array
            Phase-contrast or mCherry fluorescence image corresponding to the
            input YFP image.  The cells will be detected in this image.
        imgtype: string
            A string indicating whether the cell-detection image is a phase-
            contrast or mCherry image.  Valid inputs are 'mcherry' or 'phase'.
        threshold_op: function handle, optional
            A valid function handle for a function that returns a threshold
            from an image.  Default is 'None', which uses the imgtype to guess
            which thresholding technique will perform best.
        verbose: integer, optional
            An integer indicating the level of verbosity of the output of this
            function.  verbose=0 is the default, which prints minimal runtime
            information; verbose=2 is the maximum.
        show_plots: boolean, optional
            A flag indicating whether to create and show plots of the pipeline
            as it is being applied; default is False
        remove_outliers: boolean, optional
            A flag indicating whether the top 10 outlying pixels of 'img'
            should be removed; default is True.  This improves the performance
            of the histogram-based thresholding techniques.
        minArea: integer, optional
            Parameter used for the blob-area filtering.  Default is None, in
            which case the size will be inferred from the structuring elements
            and the imgtype.

    Returns
    -------
        YFPMean: float
            Mean YFP fluorescence of the cells in the image 'YFP'
        blobs: skimage.measure.regionprops object
            Properties of the detected cell regions in 'img'
        labelImg: 2D numpy array
            Image labeled according to the detected cells in 'img'
        fig: matplotlib figure object
            If plotting indicated, returns figure object, else None
    """
    
    ######################################################################
    # Check inputs
    ######################################################################
    
    # Check that the image shapes match
    if not np.all(YFP.shape == img.shape):
        err_str = f"Shape of input argument 'YFP' ({YFP.shape}) does not "
        err_str += f"match that of input argument 'img' ({img.shape})."
        raise ValueError(err_str)
    
    # Check that imgtype is a string...
    if not isinstance(imgtype, str):
        err_str = "Input argument 'imgtype' must be a string!"
        raise TypeError(err_str)
    # ... and that it is in the list of acceptable types
    elif imgtype.lower() not in ['mcherry', 'phase']:
        err_str = "Invalid value for inpur argument 'imgtype'. "
        err_str += "(Must be 'mCherry' or 'phase')."
        raise ValueError(err_str)
    # Make it lowercase to avoid casing issues
    imgtype = imgtype.lower()
    
    # Check that the user has given a function...
    if threshold_op is not None:
        try:
            _ = threshold_op(img)
        except: # ... or set the value to None
            if verbose:
                print_str = "There was an error using the input keyword "
                print_str += "argument 'threshold_op', using default instead!"
                print(print_str)
            threshold_op = None
            
    # If None, set the operation to the default for each image type.
    else:
        if imgtype == "mcherry":
            threshold_op = skf.threshold_otsu
        else:
            threshold_op = lambda x: skf.threshold_minimum(x) + 20
    
    # If indicated, remove the outliers from the image.
    if remove_outliers:
        frac = (img.size - 10)/img.size*100
        p99 = np.percentile(img.ravel(), frac)
        img[img>=p99] = p99
    
    ######################################################################
    # Image Processing Pipeline
    ######################################################################
    
    # Calculate the threshold
    thresh = threshold_op(img)
    if verbose:
        print(f"The threshold value using '{threshold_op.__name__}' "+
              f"is {thresh}.\n")
    
    # Create a thresholded image.
    if imgtype == 'mcherry':
        threshImg = img >= thresh # In mCherry, looking for high-fluor regions
    else:
        threshImg = img <= thresh # In phase, looking for low-fluor regions

    # Apply appropriate morphological filtering
    if imgtype == 'mcherry':
        strel1, strel2 = morph.disk(1), morph.disk(1)
        minArea = 40
    else:
        strel1, strel2 = morph.disk(2), morph.disk(5)
        minArea = np.max([len(strel1)**2, len(strel2)**2])
    
    # Apply an erosion 
    if verbose:
        print("Eroding thresholded image!")
    erodedImg = morph.erosion(threshImg, strel1)
    
    # Apply a dilation
    if verbose:
        print("Dilating eroded image!")
    dilatedImg = morph.dilation(erodedImg, strel2)
    
    # Label the image and get the region properties
    labelImg = skm.label(dilatedImg)
    blobs = skm.regionprops(labelImg)
    
    # Filter by size
    if verbose:
        print(f"Filtering by size (>{minArea}).\n")
    for blob in blobs:
        if verbose > 1:
            print(f"Blob {blob.label} has area {blob.area}")
        if blob.area <= minArea:
            labelImg[labelImg == blob.label] = 0
    print()
    
    # Calculate a new "cleanest" binary image
    cleanestImg = (labelImg > 0).astype(float)
    # Label the image and get the region properties
    labelImg = skm.label(cleanestImg)
    blobs = skm.regionprops(labelImg)
    
    # Calculate the mean YFP across cells.
    YFPMean = 0.
    for blob in blobs:
        blob.YFP = np.sum(YFP[blob.label==labelImg])/blob.area
        YFPMean += blob.YFP
        if verbose:
            print(f"The YFP of blob {blob.label} is {blob.YFP} " +
                  f"(Area: {blob.area}).")
    
    YFPMean /= len(blobs) # Normalize by the number of regions.
    
    # Print the mean!
    if verbose:
        print(f"\nThe mean YFP across ~{len(blobs)} cells is {YFPMean:.2f}.\n")
    
    
    ######################################################################
    # Plotting!
    ######################################################################
    if show_plots:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(img)
        axes[0, 0].grid(False)
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[0, 0].set_title(f"Raw {imgtype} Image", fontsize=20)
        
        sns.distplot(img.ravel(), ax=axes[0, 1], kde=False,
                     norm_hist=True)
        axes[0, 1].set_yscale('log')
        ylim = axes[0, 1].get_ylim()
        axes[0, 1].plot(2*[thresh], ylim, label="Threshold")
        axes[0, 1].set_ylim(ylim)
        axes[0, 1].legend(fontsize=14)
        axes[0, 1].set_xlabel("Fluorescence (A.U.)", fontsize=16)
        axes[0, 1].set_ylabel("Frequency",fontsize=16)
        axes[0, 1].set_title(f"Histogram of {imgtype} Fluorescence",
                            fontsize=20)
        
        axes[1, 0].imshow(threshImg)
        axes[1, 0].grid(False)
        axes[1, 0].set_xticks([])
        axes[1, 0].set_yticks([])
        axes[1, 0].set_title(f"Thresholded Image", fontsize=20)
        
        axes[1, 1].imshow(erodedImg)
        axes[1, 1].grid(False)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title(f"Eroded Image", fontsize=20)
        
        axes[2, 0].imshow(dilatedImg)
        axes[2, 0].grid(False)
        axes[2, 0].set_xticks([])
        axes[2, 0].set_yticks([])
        axes[2, 0].set_title(f"Opened Image", fontsize=20)
        
        axes[2, 1].imshow(cleanestImg)
        axes[2, 1].grid(False)
        axes[2, 1].set_xticks([])
        axes[2, 1].set_yticks([])
        axes[2, 1].set_title(f"Size-Filtered Image", fontsize=20)
        
        fig.tight_layout()
        
        return YFPMean, blobs, labelImg, fig

    else:
        return YFPMean, blobs, labelImg, None


def getYFP_AllStrains(dataDir=dataDir, verbose=1, show_plots=False):
	"""getYFP_AllStrains(dataDir=dataDir, verbose=1, show_plots=False)

	Creates a dictionary containing the mean YFP, blobs, and a labeled image of 
	the cells in a YFP fluorescence image.

	This function looks in dataDir for folders with certain strain names, loads
	the corresponding TIF images, uses the mCherry images to detect the location
	of the bacteria in the images, and then measures the YFP fluorescence of
	those pixels using the getYFP_Image function.  This function will also
	generate 6-panel figures of each of the replicates of each of the strains
	if show_plots is set to True.

	Inputs:
	=======
		dataDir		(str) Path to location of strain folders.  Optional; default
					is "Phillips_Garcia_Data/laci_full_set".
		verbose		(int) Verbosity of the function.  Set to 0 to suppress
					printing to the screen.
		show_plots	(bool) Boolean flag indicating whether to show/make figures
					showing the image processing of each replicate of each 
					strain.

	Outputs:
	========
		strainInfo	(dict) Dictionary containing mean YFP, "blobs" - detected
					regions in segmented images, and a labeled image indicating
					where bacteria have been detected.  This dictionary will
					also contain a "background" image generated from the "Auto"
					strain corresponding to the *autofluorescence* of an image
					at the wavelength of YFP.
	"""

	## Get the file paths of the Auto strain
	[autoYFP, autoMChr, autoPhase] = strainFileFinder("Auto", dataDir=dataDir)

	## Read in the first YFP image to help size the background calculation
	YFP = plt.imread(autoYFP[0])
	YFPbkgd = np.zeros((*YFP.shape, len(autoYFP)))

	## Loop through the auto replicates
	for replNo in range(len(autoYFP)):
		YFP = plt.imread(autoYFP[replNo])
		YFPbkgd[:, :, replNo] = YFP.copy()

	## Calculate the median background
	YFPbkgd = np.median(YFPbkgd, axis=2).astype(np.uint16)

	## Create the strainInfo dictionary, save the background image
	strainInfo = {}
	strainInfo['Background'] = YFPbkgd.copy()

	## Loop through the strains
	for strain in strainNames:

		if verbose >= 0:
			print(f"\nLooking at strain '{strain}'...")

		## For each strain, initialize a dictionary
		strainInfo[strain] = {}

		if verbose >= 1:
			print("Finding files...")

		## Find the file paths for each strain.
		[strYFP, strMChr, strPhase] = strainFileFinder(strain, dataDir=dataDir)

		## For each replicate of each strain...
		for replNo, YFPfile in enumerate(strYFP):

			## Initialize a dictionary (I know, nested dictionaries!)
			strainInfo[strain][replNo] = {}

			## Read the YFP and mCherry images in.
			YFP = plt.imread(YFPfile)
			mChr = plt.imread(strMChr[replNo])

			## Process them to get the bacteria locations and mean YFP
			[YFPMean, blobs,
			 labelImg, fig_h] = getYFP_Image(YFP, mChr, "mCherry",
			 								 show_plots=show_plots)

			## Save the processed info.
			strainInfo[strain][replNo]['YFPMean'] = YFPMean
			strainInfo[strain][replNo]['blobs'] = blobs
			strainInfo[strain][replNo]['Img'] = labelImg.copy()

			if verbose >= 1:
				print(f"\tReplicate {replNo} has mean YFP = {YFPMean:.5g}")
			elif verbose >= 2:
				print(f"\tReplicate {replNo} has {len(blobs)} detected blobs!")

			## If indicated, save the figures, then close (we're looping!)
			if fig_h is not None:
				savename = f"Pipeline_mCherry_{strain}{replNo}"
				saveFigure(fig, savename, figDir=os.path.join(dataDir, strain))
				plt.close(fig_h)

	return strainInfo


def calcFoldChange(strainInfo, verbose=1):
	"""calcFoldChange(strainInfo, verbose=1)

	Uses the strainInfo dictionary to calculate the fold-change in YFP fluores-
	cence compared to the Delta strain (no LacI repressor at all; max YFP).

	Inputs:
	=======
		strainInfo	(dict) Dictionary containing mean YFP, "blobs" - detected
					regions in segmented images, and a labeled image indicating
					where bacteria have been detected.  This dictionary will
					also contain a "background" image generated from the "Auto"
					strain corresponding to the *autofluorescence* of an image
					at the wavelength of YFP.
		verbose		(int) Verbosity of the function.  Set to 0 to suppress
					printing to the screen.

	Outputs:
	========
		strainInfo	(dict) Same as input but with the 'deltaMean' field and
					'FoldChange' fields added to each strain.
	"""

	## Get the background
	YFPbkgd = strainInfo['Background']

	## Calculate the mean YFP of the Delta strain (minus the background!)
	deltaMean = np.mean([strainInfo['Delta'][replNo]['YFPMean']-np.mean(YFPbkgd)
		for replNo in range(len(strainInfo['Delta']))])

	## Save the Delta strain mean
	strainInfo['deltaMean'] = deltaMean

	if verbose >= 1:
		print(f"\nThe average YFP with no repressors is {deltaMean:.5g}")
		print(f"The average YFP background is {np.mean(YFPbkgd):.5g}")

	## Loop through the strains
	for strain in strainInfo:
		if strain not in strainNames:
			continue  ## Skip fields not corresponding to strains
		if strain in ['Auto', 'Delta']:
			continue  ## Skip the two "control" strains

		## Get mean across replicates of strain...
		YFPMean = np.mean([strainInfo[strain][replNo]['YFPMean']
			for replNo in range(len(strainInfo[strain]))])

		if verbose >= 1:
			print(f"\nThe average YFP in {strain} is {YFPMean:.5g}")

		## Save the mean across replicates
		strainInfo[strain]['YFPMean'] = YFPMean

		## Save the fold-change in YFP fluorescence across replicates.
		## Note that we adjust for background then compare to deltaMean
		strainInfo[strain]['FoldChange'] = (YFPMean-np.mean(YFPbkgd))/deltaMean

		if verbose >= 1:
			print(f"The fold change expression of YFP is "+
				f"{strainInfo[strain]['FoldChange']:.5g}")

	return strainInfo


def saveFigure(fig, figName, formats=['pdf', 'png'], dpi=600, figDir="."):
	"""saveFigure(fig, figName, formats=['pdf', 'png'], dpi=600, figDir='.')

	Utility function for quickly saving figures in multiple formats.

	Inputs:
	=======
		fig 		(figure handle) Figure to be saved
		figName		(str) Name of figure to be saved (NO file extension!)
		formats		(list) List of formats in which to save the figure. 
					Optional; default is pdf and png
		dpi 		(int) Pixels per inch (image resolution).  Optional;
					default is 600.
		figDir		(str) Folder in which to save the images.  Optional; 
					default is the current working directory.
	"""

	if not isinstance(figName, str):
		raise ValueError(f"Input argument figName={figName} must be a string!")
	if not os.path.isdir(figDir):
		raise FileError("Keyword argument figDir={figDir} must be a "+
			"valid directory!")
	if not isinstance(dpi, int):
		raise ValueError(f"Keyword argument dpi={dpi} must be an integer!")

	for form in formats:
		figPath = os.path.join(figDir, figName + "." + form)
		try:
			fig.savefig(figPath, format=form, dpi=dpi)
		except:
			print(f"Could not save {figPath} as format {form}!")

	return


if __name__ == "__main__":
	print("\n\n" + 52*"=" + "\n\tTESTING IMAGE PROCESSING PACKAGE!\n" +
		52*"=" + "\n")

	plt.close('all')

	anyError = False
	if True:
		print("Testing strainFileFinder...")
		try:
			strainFileFinder(2)
			print("\tError Check 1: Failed!")
			anyError = True
		except:
			print("\tError Check 1: Passed!")
		try:
			strainFileFinder("myStrain")
			print("\tError Check 2: Failed!")
			anyError = True
		except:
			print("\tError Check 2: Passed!")
		try:
			strainFileFinder('Auto', dataDir=2)
			print("\tError Check 3: Failed!")
			anyError = True
		except:
			print("\tError Check 3: Passed!")
		try:
			strainFileFinder('Auto', dataDir="myDir")
			print("\tError Check 4: Failed!")
			anyError = True
		except:
			print("\tError Check 4: Passed!\n")

		try:
			testStrain = 'RBS1'
			testYFPList = [os.path.join(dataDir, testStrain, 
				f"YFP_regulated_{ii+1}.tif")for ii in range(4)]
			testmChrList = [os.path.join(dataDir, testStrain, 
				f"mCherry_constitutive_{ii+1}.tif") for ii in range(4)]
			testPhaseList = [os.path.join(dataDir, testStrain, 
				f"Phase_{ii+1}.tif") for ii in range(4)]
			# print(testYFPList, testmChrList, testPhaseList)
			outYFP, outmChr, outPhase = strainFileFinder(testStrain)
			# print(outYFP, outmChr, outPhase)
			assert outYFP == testYFPList, "Here 1"
			assert outmChr == testmChrList
			assert outPhase == testPhaseList
			print("\tOutput Check: Passed!")
		except:
			print("\tOutput Check: Failed!\n")
			anyError = True

	if True:
		print("\nTesting channelPlotter...")
		fig, axes = channelPlotter(outmChr)
		fig.tight_layout()

	if True:
		print("\nTesting channelPixelDistPlotter...")
		fig, axes = channelPixelDistPlotter(outPhase[0])
		fig.tight_layout()

	if True:
		print("\nTesting getYFP_Image...")
		YFPImg = plt.imread(outYFP[0])
		mChrImg = plt.imread(outmChr[0])
		[YFPMean, blobs,
		 labelImg, fig_h] = getYFP_Image(YFPImg, mChrImg, "mCherry",
		 							show_plots=True)
		fig_h.tight_layout()

	if True:
		print("\nTesting getYFP_AllStrains...")

		strainInfo = getYFP_AllStrains()

	if True:
		print("\nTesting calcFoldChange...")

		strainInfo = calcFoldChange(strainInfo)

	if anyError:
		raise ValueError("Something didn't pass its checks!")

	plt.show()
