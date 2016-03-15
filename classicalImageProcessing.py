#!/usr/bin/env python

#
# A "classical" image processing pipeline to detect coins on a surface
#

from  __future__ import print_function, division
import skimage.filters as skfilt
import skimage.io as skio
import skimage.measure as skmeas
import skimage.morphology as skmo
import skimage.transform as sktrans
import skimage.draw as skdraw
import skimage.feature as skfeat
import skimage.color as skcolor
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi


try:
    import jm_packages.image_processing.filters as jmfilt
except:
    print("Warning: JM_PACKAGES NOT PRESENT ON THIS MACHINE")
    print("Please contact the script author")
    jmfilt = None

def load_data(filename):
    """
    Probably just an alias to skimage.io.imread, unless we need to
    do other basic pre-processing
    """
    return skio.imread(filename)


def process_file(filename):
    """
    Main processing function; loads image file, runs some simple
    filters, and returns labelled image array
    """
    data = load_data(filename)
    print("Loaded data", data.shape)
    # Grayscale
    filt = data.mean(axis=-1)
    # Basic threshold
    print("Running basic thresholding...")
    bw = filt > skfilt.threshold_li(filt)
    labels = skmeas.label(bw)
    return labels

def show_results(filename, labels):
    """
    Load the data and overlay the results labels
    """
    data = load_data(filename)
    plt.figure()
    plt.imshow(data)
    Nlabs = labels.max()
    if Nlabs < 100:
        for l in range(1, Nlabs+1):
            plt.contour(labels==l, levels=[0.5,], colors=[plt.cm.jet((l%9)/8)])
    else:
        plt.imshow(np.ma.masked_where(labels==0, labels),alpha=0.4)

def test_sample1():
    ROOT = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(ROOT, "samples", "coins1.jpg")
    data = load_data(filename)
    data = sktrans.rescale(data, 0.1)
    plt.figure()
    plt.title("Original data")
    plt.imshow(data)
    plt.figure()
    filt = data[...,0].astype(float)#.mean(axis=-1)
    filt-=filt.min()
    filt/=filt.max()
    plt.imshow(filt, cmap='jet')
    plt.title("Red channel")

    edges = skfilt.sobel(skfilt.gaussian_filter(filt, 4.0))
    plt.figure()
    plt.title("Edges (Sobel)")
    plt.imshow(edges)

    markers = np.zeros(filt.shape, dtype=int)
    med = np.median(filt)
    mad = np.median(np.abs(filt - med))
    markers[filt <  med] = 1
    markers[filt >  (med + 3*mad)] = 2
    ws = skmo.watershed(edges, markers)
    plt.figure()
    plt.title("Watershed using edges")

    plt.imshow(filt, cmap='gray')
    plt.imshow(np.ma.masked_where(ws==0, ws), alpha=0.4)

    plt.figure()
    plt.imshow(skfilt.median(filt, np.ones((5,5), dtype=bool)))
    plt.title("Median filtered image")

    # Let's try scale space representation
    scales=np.arange(5, 40)
    if jmfilt:
        lg = jmfilt.scale_space_LoG(filt, scales=scales)
        med,mad = jmfilt.getMedMad(lg)
        bw = lg > (med + 3*mad)     # Automatic thresholding based on
                                    # median absolute deviation
        # Method 1
        ## Have to use maximum filter, one scale at a time?
        #maxim = np.zeros_like(lg)
        #for i,s in enumerate(scales):
        #    lgnow = np.zeros_like(lg)
        #    lgnow[...,i] = lg[...,i]
        #    lgmax = ndi.maximum_filter(lg, footprint=skmo.ball(s))
        #    maxim = np.maximum(lgmax, maxim)
        # Method 2
        # Find local maxima, then operate on peaks to suppress
        lgmax = ndi.maximum_filter(lg, size=3)
        peaks = np.array(((lgmax == lg)&bw).nonzero()).T
        peak_ints = lg[peaks.T.tolist()]
        # Now sort the peaks by the intensities
        order = np.argsort(peak_ints)[::-1]
        peak_ints = peak_ints[order]
        peaks = peaks[order]
        # Now suppress
        active = np.ones(peaks.shape[0], dtype=bool)
        for i, p in enumerate(peaks):
            if not active[i]:
                continue
            # Silence any nearby peaks
            dists = np.sqrt(np.sum( (peaks[:,:2]-p[:2])**2, axis=-1))
            nearby = dists < (2*scales[p[2]])
            nearby[i] = False   # Ignore self
            active[nearby] = False
        suppressed = peaks[~active]
        peaks = peaks[active]
        print("Peaks after non-maximal suppression:", len(peaks))
        plt.imshow(filt, cmap='gray')
        #plt.scatter(peaks[:,1], peaks[:,0], s=10*np.array(peaks[:,2]),
        #    marker='o', color="y", facecolor="none")
        #plt.scatter(suppressed[:,1], suppressed[:,0],
        #    s=10*np.array(suppressed[:,2]),
        #    marker='o', color="r", facecolor="none")
        # Add circles instead to get proper sizes
        ax = plt.gca()
        for p in peaks:
            c = plt.Circle(p[[1,0]], scales[p[2]], color='y', fill=False)
            ax.add_artist(c)
        for p in suppressed:
            c = plt.Circle(p[[1,0]], scales[p[2]], color='r', fill=False)
            ax.add_artist(c)

        plt.title("Scale space peaks")

    # Lastly, good old Hough
    bw_edges = edges > skfilt.threshold_li(edges)
    bw_edges = skmo.skeletonize(bw_edges)
    bw_edges = skmo.binary_dilation(bw_edges)
    hough_radii = scales
    hough_res = sktrans.hough_circle(bw_edges, hough_radii)

    centers = []
    accums = []
    radii = []
    for radius, h in zip(hough_radii, hough_res):
        num_peaks = 20
        peaks = skfeat.peak_local_max(h, num_peaks=num_peaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    # Draw the most prominent 5 circles
    image = skcolor.gray2rgb(filt)
    print(image.shape)
    for idx in np.argsort(accums)[::-1][:20]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = skdraw.circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = (220, 20, 20)
    plt.figure()
    plt.imshow(filt, cmap='gray')
    plt.imshow(np.ma.masked_where(bw_edges==0, bw_edges), cmap='hsv',
        alpha=0.4)
    plt.title("Binary edges (input to hough)")
    plt.figure()
    plt.imshow(image)
    plt.title("Hough circle transform results")

    plt.show()

    return
    labels = process_file(filename)
    show_results(filename, labels)
    plt.show()

if __name__ == '__main__':
    test_sample1()
