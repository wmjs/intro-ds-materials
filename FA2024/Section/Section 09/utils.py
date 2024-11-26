import numpy as np
import matplotlib.pyplot as plt

from IPython import display

from scipy.stats import norm

from scipy.signal import find_peaks

def plot_cross_sect_with_threshold(img, threshold1, threshold2):
    def plot_img(image, plot_title):
        plt.imshow(image, cmap='gray')
        plt.title(plot_title) 
        plt.clim(0,255)
        plt.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 

    img1 = img.copy()
    img1[threshold1 < img] = 255

    img2 = img.copy()
    img2[np.logical_or(img<threshold1, threshold2<img)] = 255

    img3 = img.copy()
    img3[img < threshold2] = 255

    # plots
    plt.subplot(2, 2, 1)
    plot_img(img, 'full image')

    plt.subplot(2, 2, 2)
    plot_img(img1, 'pixel values 0 to %d' % threshold1)

    plt.subplot(2, 2, 3)
    plot_img(img2, 'pixel values %d to %d' % (threshold1, threshold2))

    plt.subplot(2, 2, 4)
    plot_img(img3, 'pixel values %d to 255' % threshold2)

    plt.show()


def initialize(X, num_clust):
    hist, bins = np.histogram(X, bins=255)
    bins = bins.astype(int)
    
    # Get the peaks in the histogram
    peaks, prop = find_peaks(hist, height=0)
    mu = np.sort(peaks[np.argsort(prop['peak_heights'])[-1:-(num_clust+1):-1]])
    
    # Set initial variance
    var = [((bins[-1]- 0)/num_clust)**1 for i in range(num_clust)]
    var = [200 for i in range(num_clust)]
    var[0] = 10
    
    # assigning same probability (sum up to one) while we don't have 
    # any information on the prior probabilty of each cluster
    cluster_prior = [0.6/(num_clust-1) for i in range(num_clust)]
    cluster_prior[0] = 0.4

    return mu, var, cluster_prior



def plot_gmm_steps(X, mu, var, cluster_prior, error, max_iter):
    # Plots the histogram along with the cluster and mixture Gaussian distributions, 
    # and the error for each iteration
    # Inputs:
    #      H: previously estimated image histogram
    #      mu: claster means (1D array)
    #      var: cluster variances (1D array)
    #      cluster_prior: cluster prior probabilities (1D array)
    # Retuen:
    #       None
    H = np.histogram(X, bins=255)

    plt.figure(figsize=(10,3))
    plt.subplot(1, 2, 1)

    mixture = np.zeros(np.shape(H[1]))
    # plot individual Guassian and update mixture
    for i in range(len(mu)):
        gauss_comp_pdf = cluster_prior[i]*norm.pdf(H[1], loc=mu[i],scale=np.sqrt(var[i]))
        plt.plot(H[1], gauss_comp_pdf, '--', label=f'cluster {i+1}')
        plt.legend()
        mixture += gauss_comp_pdf
    
    # Plot mixture
    plt.plot(H[1], mixture, label=f'GMM')
    plt.legend()
    
    # Plot image histogram
    counts = H[0]
    bins = H[1]
    plt.hist(bins[:-1], bins, weights=counts, density='True',color='gainsboro')
    plt.ylim(0,0.07)
    plt.title(f'GMM, iter {len(error)}')
    plt.xlabel('Intensity Values')

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(error))+1, error, '.-')
    plt.xlim([1,max_iter]); plt.xticks(np.arange(max_iter)+ 1)
    plt.title('Log Likelihood')
    plt.xlabel('Iterations')
    
    display.clear_output(wait=True)
    display.display(plt.gcf())

    plt.close()


def plot_original_image(Image, slice_idx):
    # Original image
    plt.imshow(Image[:,:,slice_idx],cmap="gray")
    plt.colorbar(orientation='horizontal')
    _ = plt.title('Original Image')
    plt.clim(0,255)
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    
def plot_segmented_image(Segmented_image, slice_idx):
    num_clusters = len(np.unique(Segmented_image))
    plt.imshow(Segmented_image[:,:,slice_idx],cmap="gray")
    plt.colorbar(ticks=list(range(num_clusters)), orientation='horizontal')
    _ = plt.title(f'Segmented Image (clusters = {num_clusters})')
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 