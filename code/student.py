import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int, transform, measure
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter

def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    plt.imshow(image)
    plt.scatter(x, y, c='b', s=40, marker='x')
    plt.title('Interest Points')
    plt.show()
    

def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_interest_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''
    # UNCOMMENT TO USE THE MSER INTEREST POINT ALGORITHM!!!
    return mser_interest_points(image)

    image = image.T
    image = img_as_int(image)
    

    dx = np.gradient(image, axis=1)
    dy = np.gradient(image, axis=0)

    dx2 = gaussian_filter(dx**2, sigma=2)
    dy2 = gaussian_filter(dy**2, sigma=2)
    dxy = gaussian_filter(dx * dy, sigma=2)
    

    k = 0.04
    harris_response = dx2 * dy2 - dxy**2 - k * (dx2 + dy2)**2

    threshold = 0.15 * np.max(harris_response)
    xs, ys = np.where(harris_response > threshold)

    exclude_near_edges_mask = (xs >= feature_width/2) & (xs < image.shape[1] - feature_width/2) & (ys >= feature_width/2) & (ys < image.shape[0] - feature_width/2)
    xs = xs[exclude_near_edges_mask]
    ys = ys[exclude_near_edges_mask]
    
    return np.array(xs), np.array(ys)

def mser_interest_points(image, delta=.2):
    interest_points = []
    thresholds = np.arange(np.min(image), np.max(image), delta)
    
    for threshold in thresholds:
        image_bin = image > threshold
        
        labels = measure.label(image_bin, background=0)
        region_properties = measure.regionprops(labels)
        
        for prop in region_properties:
            y0, x0 = prop.centroid
            interest_points.append((int(x0), int(y0)))
            
    
    unique_points = list(set(interest_points))
    
    xs = np.array([point[0] for point in unique_points])
    ys = np.array([point[1] for point in unique_points])
    
    return xs, ys

def get_features(image, x, y, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    histogram_vectors = []
    image_orig = img_as_int(image)
    
    # pyramid = list(transform.pyramid_gaussian(image_orig, max_layer=3, downscale=2))
    pyramid = [image_orig]
    for image_gaus in pyramid:
        gradient_mag, gradient_orient = get_gradients(image_gaus)
        for x_pt, y_pt in zip(x, y):
            histogram = calculate_histogram_pt(gradient_mag, gradient_orient, x_pt, y_pt, feature_width)
            
            if max(histogram) > 0:
                histogram_vectors.append(histogram)

    return np.array(histogram_vectors)

def get_gradients(image):
    smoothed = gaussian_filter(image, sigma=2)
    
    gradient_x = filters.sobel_h(smoothed)
    gradient_y = filters.sobel_v(smoothed)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi) % 360

    return magnitude, orientation

def calculate_histogram_pt(gradient_mag, gradient_orient, x, y, feature_width):
    feature_vector = []
    cell_size = feature_width // 4

    for i in range(feature_width // cell_size):
        for j in range(feature_width // cell_size):
            x0 = int(x - feature_width // 2 + i * cell_size)
            y0 = int(y - feature_width // 2 + j * cell_size)
            x1 = int(x0 + cell_size)
            y1 = int(y0 + cell_size)
            
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, gradient_mag.shape[1]), min(y1, gradient_mag.shape[0])

            cell_mags = gradient_mag[y0:y1, x0:x1]
            cell_orients = gradient_orient[y0:y1, x0:x1]

            histogram = [0] * 8
            for mag, orient in zip(cell_mags.flat, cell_orients.flat):
                bin_index = int(orient // 45) % 8
                histogram[bin_index] += mag
                
            if np.linalg.norm(histogram) > 0:
                histogram /= np.linalg.norm(histogram)
            feature_vector.extend(histogram)
            
    feature_vector = np.array(feature_vector, dtype=np.float32)
    
    return feature_vector


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
    B = 2 * np.dot(im1_features, im2_features.T)
    F1 = np.sum(np.square(im1_features), axis=1, keepdims=True)
    F2 = np.sum(np.square(im2_features), axis=1, keepdims=True).T
    D = np.sqrt(F1 + F2 - B)
    
    sorted_indicies = np.argsort(D, axis=1)
    neighbor1 = sorted_indicies[:, 0]
    neighbor2 = sorted_indicies[:, 1]
    
    nndr = D[np.arange(D.shape[0]), neighbor1] / D[np.arange(D.shape[0]), neighbor2]
    matches_mask = nndr < 0.9
    matches = np.column_stack([np.arange(D.shape[0])[matches_mask], neighbor1[matches_mask]])
    confidences = 1 - nndr[matches[:, 0]]

    return matches, confidences
