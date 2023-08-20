import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import psana as ps

width = 1.5
length = 5

plt.rcParams['figure.figsize'] = (7.5, 4.5)
plt.rcParams['font.size'] = 12
plt.rcParams['xtick.major.width'] = width
plt.rcParams['ytick.major.width'] = width
plt.rcParams['xtick.major.size'] = length
plt.rcParams['ytick.major.size'] = length
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

class plotting:
    def __init__(self):
        pass
    
class post_analysis:
    def __init__(self):
        pass
    
    def expfunc(x, k):
        if k.size == 1:
            return np.exp(-k*x)
        elif k.size > 1:
            f = np.empty((time.size, k.size))
            for ii in range(k.size):
                f[:,ii] = np.exp(-k[ii]*x)
            return f

    def expfunc_heaviside(x, k):
        if k.size == 1:
            return np.exp(-k*x)*np.heaviside(x, 0.5)
        elif k.size > 1:
            f = np.empty((x.size, k.size))
            for ii in range(k.size):
                f[:,ii] = np.exp(-k[ii]*x)*np.heaviside(x, 0.5)
            return f

    def gaussfunc(x, center, sigma):
        return np.exp((-(x - center)**2)/(2*sigma**2))

    def irfconv(time, k, irf_center, irf_sigma, exp_func):
        dt = min(np.diff(time))
        conv_tgrid = np.arange(min(time), max(time)*1.2, dt)

        expmat = exp_func(conv_tgrid, k)
        gaussmat = np.empty_like(expmat)

        if k.size == 1:
            gaussmat[:] = gaussfunc(conv_tgrid, irf_t0, irf_sigma)
        elif k.size > 1:
            for ii in range(k.size):
                gaussmat[:,ii] = gaussfunc(conv_tgrid, irf_t0, irf_sigma) # for fftconvolve, we need expmat and gaussmat to have same dimensions

        pre0 = conv_tgrid[conv_tgrid < 0].size

        conv_vec = scipy.signal.fftconvolve(expmat, gaussmat, axes = 0)
        C = conv_vec[pre0:(len(time)+pre0)]*dt

        return C

    def svdplot(xval, yval, data, ncomp):
        U, S, V = np.linalg.svd(data)

        fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (8,4))

        offsetU = 0
        offsetbaseU = np.max(np.abs(U[:,0:ncomp]))
        offsetV = 0
        offsetbaseV = np.max(np.abs(V[0:ncomp,:]))
        offsetlistU = []
        offsetlistV = []
        SVDindex = []

        for ii in range(ncomp):
            SVDindex.append(ii+1)
            ax[0].plot(xval, U[:,ii] + offsetU, linewidth = 2)
            ax[1].scatter(SVDindex[ii], S[ii])
            ax[2].plot(yval, V[ii,:] + offsetV, linewidth = 2)
            offsetlistU.append(offsetU)
            offsetlistV.append(offsetV)
            offsetU -= offsetbaseU
            offsetV -= offsetbaseV

        width = 1.5

        for plot in ax:
            for axis in ['top', 'bottom', 'left', 'right']:
                plot.spines[axis].set_linewidth(width)

        ax[0].set_yticks(offsetlistU)
        ax[0].set_yticklabels(SVDindex)
        ax[0].set_title('Left Singular Vectors')
        ax[0].set_xlim([min(xval), max(xval)])
        ax[1].set_yscale('log')
        ax[1].set_xlim([min(SVDindex)-1, max(SVDindex)+1])
        ax[1].set_xticks(SVDindex)
        ax[1].set_xticklabels(SVDindex)
        ax[1].set_title('Scree Plot')
        ax[2].set_yticks(offsetlistV)
        ax[2].set_yticklabels(SVDindex)
        ax[2].set_title('Right Singular Vectors')
        ax[2].set_xlim([min(yval), max(yval)])
        
    def svdreconstruct(data, ncomp):
        U, S, V = np.linalg.svd(data)

        data_reconstruct = U[:,0:ncomp]@np.diag(S[0:ncomp])@V[0:ncomp,:]
        
        print('SVD Reconstruction performed with {} components'.format(ncomp))

        return data_reconstruct