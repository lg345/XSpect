import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
#import psana as ps

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
    
class analysis_functions:
    def __init__(self):
        pass
    
    def expfunc(self, x, k, amp = [], x0 = 0):
        ## returns a vector (or matrix of column vectors, depending on the length of k) containing an exponential decay evalutated over x values x with rate constant(s) k
        if not isinstance(k, list):
            k = [k]
        k = np.abs(k) # always takes positive values and enforces exp decay
        f = np.empty((len(x), len(k)))
        for i in range(len(k)):
            f[:,i] = np.exp(-k[i]*(x-x0))
        return f
    
    def expfunc_heaviside(self, x, k, amp = [], x0 = 0):
        ## returns a vector (or matrix of column vectors) containing an exponential decay with rate constant(s) k multiplied by the heaviside step function (x < 0, y = 0; x = 0, y = 0.5; x > 0, y = 1) evaluated over x
        if isinstance(k, list):
            k = np.array(k)
        if isinstance(k, float) or isinstance(k, int):
            k = np.array([k])

        if isinstance(amp, list):
            amp = np.array(amp)
        if isinstance(amp, float) or isinstance(amp, int):
            amp = np.array([amp])
        if not amp.size > 0:
            amp = np.ones_like(np.array(k))
            
        
        k = np.abs(k)
        f = np.empty((len(x), len(k)))
        for i in range(len(k)):
            f[:,i] = amp[i]*np.exp(-k[i]*(x-x0))*np.heaviside((x-x0), 0.5) ## h(x) = 0 (x <= 0), h(x) = 1 (x > 0)
        return f
    
    def gaussfunc(self, x, center, sigma):
        return np.exp((-(x - center)**2)/(2*sigma**2))

    def gaussfunc_norm(self, x, center, sigma):
        return (1/(sigma*np.sqrt(2*np.pi)))*np.exp((-(x - center)**2)/(2*sigma**2))

    def irfconv(self, x, k, center, sigma, amp = []):
        
        ## numerical convolution between heaviside*exponential and gaussian IRF
        
        if isinstance(k, list):
            k = np.array(k)
        if isinstance(k, float) or isinstance(k, int):
            k = np.array([k])

        if isinstance(amp, list):
            amp = np.array(amp)
        if isinstance(amp, float) or isinstance(amp, int):
            amp = np.array([amp])
        if not amp.size > 0:
            amp = np.ones_like(np.array(k))

        dx = min(np.diff(x))
        conv_xgrid = np.arange(min(x), max(x)*1.2, dx/10)
        cdx = min(np.diff(conv_xgrid))

        expmat = self.expfunc_heaviside(conv_xgrid, k, amp)
        gaussmat = np.empty_like(expmat)

        ## for fftconvolve, we need expmat and gaussmat to have same dimensions

        if sigma == 0:
            ## if linewidth is given as 0, then convolve with a delta fxn (normalization factor cdx set to 1)
            for i in range(len(k)):
                gaussmat[:,i] = scipy.signal.unit_impulse(gaussmat[:,i].shape, idx = np.argmin(abs(conv_xgrid-center)))
            cdx = 1
        else:
            for i in range(len(k)):
                gaussmat[:,i] = self.gaussfunc_norm(conv_xgrid, center, sigma)
    #     elif sigma < 0:
    #         print('Error: cannot have negative width parameter')
    #         print('Using absolute value of given input')
    #         for i in range(len(k)):
    #             gaussmat[:,i] = gaussfunc_norm(conv_xgrid, center, np.abs(sigma))

        pre0 = conv_xgrid[conv_xgrid < 0].size

        conv_vec = scipy.signal.fftconvolve(expmat, gaussmat, axes = 0)
        C = conv_vec[pre0:(len(conv_xgrid)+pre0)]*cdx

        C_interp = scipy.interpolate.interp1d(conv_xgrid, C, axis = 0)(x)

        return C_interp
    
    def irfconv_ana(self, x, k, center, sigma, amp = []):
        
        ## returns analytical convolution between heaviside*exponential and gaussian
        
        if isinstance(k, list):
            k = np.array(k)
        if isinstance(k, float) or isinstance(k, int):
            k = np.array([k])

        if isinstance(amp, list):
            amp = np.array(amp)
        if isinstance(amp, float) or isinstance(amp, int):
            amp = np.array([amp])
        if not amp.size > 0:
            amp = np.ones_like(np.array(k))

        k = np.abs(k)
        f = np.empty((len(x), len(k)))

        if sigma > 0:
            for i in range(len(k)):
                a1 = amp[i]/2
                a2 = np.exp(-k[i]*(x-center))
                a3 = np.exp(0.5*(k[i]*sigma)**2)
                a4 = 1 + scipy.special.erf((x - center - k[i]*(sigma**2))/(np.sqrt(2)*sigma))
                f[:,i] = a1*a2*a3*a4
        elif sigma == 0:
            f = self.expfunc_heaviside(x, k, amp = amp, x0 = center)
        else:
            print('Error: IRF linewidth must be positive')
            return
        return f
    
    def kmatsolver(self, kmatrix, x, k, X0, center, sigma, irf_option = 'numerical', printopt = True):
        
        ## Berberan-Santos, M. N. and Martinho, J. M. G. The integration of kinetic rate equations by matrix methods. J. Chem. Ed. 1990, 67, 375
        
        ## For solving a system of rate equations that are purely composed of unimolecular (first-order) steps, the analytical solution always exists and can be found using the rate constant (k) matrix and its eigenvectors/values
        
        ## dX1/dt = k11*X1 + k12*X2
        ## dX2/dt = k21*X1 + k22*X2
        ## 
        ## Recast in matrix form:
        ##
        ## [dX1/dt]    [k11 k12] [X1]
        ## [dX2/dt] =  [k21 k22] [X2]
        ## 
        
        if isinstance(k, list):
            k = np.array(k)
        if isinstance(k, float) or isinstance(k, int):
            k = np.array([k])
        
        ## kmatrix is easiest to define as a lambda function that accepts a list of rate constants and uses that to populate an array (kmat)
        
        kmat = kmatrix(k)

        if kmat.shape == (1,):
            kmat = np.expand_dims(kmat, axis = 1)

        if printopt:
            print('Printing k Matrix:')
            print(kmat)
        
        ## solve eigenvalues/eigenvectors of k matrix; the eigenvalues are the rates and the eigenvectors inform what linear combination of exponentials leads to the solution of the system of equations

        eigval, eigvec = np.linalg.eig(kmat)

        ## a is a set of constants dependent on the eigenvectors and initial amplitudes
        
        if len(X0) == 1:
            a = np.linalg.inv(eigvec)*X0
        elif len(X0) > 1:
            a = np.linalg.inv(eigvec)@X0

        ## evaluate exponential fxns (with or without gaussian IRF convolution) and store as column vectors in v
        
        if irf_option == 'numerical':
            v = self.irfconv(x, eigval, center, sigma)
        elif irf_option == 'analytical':
            v = self.irfconv_ana(x, eigval, center, sigma)
        elif irf_option == 'none':
            v = self.expfunc_heaviside(x, eigval)
        else:
            print('irf_option can only take "numerical", "analytical", or "none"')

        ## Xt is the final concentration matrix
        
        Xt = np.transpose(eigvec@np.transpose(a*v))

        return Xt 
    
class post_analysis(analysis_functions):
    def __init__(self):
        pass

    def svdplot(self, xval, yval, data, ncomp, figdim=(8,4)):
        U, S, V = np.linalg.svd(data)

        fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = figdim)

        offsetU = 0
        offsetbaseU = np.max(np.abs(U[:,0:ncomp])) * 2.0
        offsetV = 0
        offsetbaseV = np.max(np.abs(V[0:ncomp,:])) * 2.0
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
        
    def svdreconstruct(self, data, ncomp):
        U, S, V = np.linalg.svd(data)

        data_reconstruct = U[:,0:ncomp]@np.diag(S[0:ncomp])@V[0:ncomp,:]
        
        print('SVD Reconstruction performed with {} components'.format(ncomp))

        return data_reconstruct
    
    def parse_theta(self, k = [], center = [], sigma = [], amplitudes = []):
        theta_parser = {'k': k, 'center': center, 'sigma': sigma, 'amplitudes': amplitudes}

        return theta_parser

    def construct_theta(self, theta_parser):
        theta = []
        for count, key in enumerate(list(theta_parser.keys())):
            theta = theta + theta_parser[key]
        return np.array(theta)

    def read_theta(self, theta, theta_parser):

        k_guess = []
        center_guess = []
        sigma_guess = []
        amplitude_guess = []

        nk = len(theta_parser['k'])
        ncenter = len(theta_parser['center'])
        nsigma = len(theta_parser['sigma'])
        namplitudes = len(theta_parser['amplitudes'])

        k_guess = theta[0:nk]
        center_guess = theta[nk:nk+ncenter]
        sigma_guess = theta[nk+ncenter:nk+ncenter+nsigma]
        amplitudes_guess = theta[nk+ncenter+nsigma:nk+ncenter+nsigma+namplitudes]

        parsed = self.parse_theta(k = k_guess, center = center_guess, sigma = sigma_guess, amplitudes = amplitudes_guess)

        return parsed
    
    def varproj(self, kmatrix, x, k, X0, center, sigma, data):
        C = self.kmatsolver(kmatrix, x, k, X0, center, sigma, printopt = False)

        C_inv = np.linalg.pinv(C)

        Et = C_inv@data

        SimA = C@Et

        return C, np.transpose(Et), SimA

    def targetobjective(self, theta, x, kmatrix, X0, theta_parser, data):
        theta_dict = self.read_theta(theta, theta_parser)
        C_guess, DAS, SimA = self.varproj(kmatrix, x, theta_dict['k'], X0, theta_dict['center'], theta_dict['sigma'], data)

        res = data - SimA

        resl = res.ravel()

        return resl

    def targetanalysis_run(self, data, x, kmatrix, k_in, center_in, sigma_in, X0_in, y = [], bounds_dict = None):
        
        theta_dict = self.parse_theta(k = k_in, center = center_in, sigma = sigma_in)
        theta_in = self.construct_theta(theta_dict)
        
        if bounds_dict == None:
            ## set default parameter bounds
            bounds_dict = {'lb': {}, 'ub':{}}
        
            for count, key in enumerate(list(theta_dict.keys())):
                bounds_dict['lb'][key] = []
                bounds_dict['ub'][key] = []
                if key == 'k':
                    default_bound = [0, np.inf]
                elif key == 'center':
                    default_bound = [-np.inf, np.inf]
                elif key == 'sigma':
                    default_bound = [0, np.inf]
                elif key == 'amplitudes':
                    default_bound = [-np.inf, np.inf]
                for i in range(len(theta_dict[key])):
                    bounds_dict['lb'][key].append(default_bound[0])
                    bounds_dict['ub'][key].append(default_bound[1])
            
        lower = []
        upper = []
        for count, key in enumerate(list(bounds_dict['lb'].keys())):
            lower += bounds_dict['lb'][key]
            upper += bounds_dict['ub'][key]
            
        constraints = (lower, upper)
        
        res_lsq = scipy.optimize.least_squares(self.targetobjective, theta_in, args = (x, kmatrix, X0_in, theta_dict, data), method = 'trf', bounds = constraints)

        theta_out = self.read_theta(res_lsq.x, theta_dict)
        if 'k' in theta_out.keys():
            theta_out['k'][::-1].sort()

        C_fit, E_fit, A_fit = self.varproj(kmatrix, x, theta_out['k'], X0_in, theta_out['center'], theta_out['sigma'], data)

        print('Fit Parameters:')
        print(theta_out)
        print('')
        print('Cost:')
        print(res_lsq.cost)

        fig, ax = plt.subplots(ncols = 2, nrows = 1)

        ax[0].plot(x, C_fit)
        ax[0].set_title('Concentration')
        ax[0].set_xlabel('Time')
        ax[0].set_ylabel('Amplitude')
        ax[0].set_xlim((min(x), max(x)))

        if isinstance(y, np.ndarray):
            ax[1].plot(y, E_fit)
            ax[1].set_xlabel('')
            ax[1].set_xlim((min(y), max(y)))
        elif not y:
            ax[1].plot(E_fit)
        ax[1].set_title('EAS')
    #     ax[1].set_ylabel('Amplitude')

        fig.suptitle('Target Analysis Results', fontsize = 16)

        for plot in ax:
            for axis in ['top', 'bottom', 'left', 'right']:
                plot.spines[axis].set_linewidth(width)
        
        # calculate standard errors (se) and 95% confidence intervals (ci) from jacobian
        J = res_lsq.jac
        res = res_lsq.fun
        n = res.shape[0]
        p = J.shape[1]
        v = n - p
        rmse = np.linalg.norm(res, ord = 2)/np.sqrt(v)
        cov = np.linalg.inv(J.T@J)*rmse**2
        se = np.sqrt(np.diag(cov))
        setattr(res_lsq, 'se', se)
        
        a = 0.05
        delta = se*scipy.stats.t.ppf(1 - a/2, v)
        ci = np.array([res_lsq.x-delta, res_lsq.x+delta])
        setattr(res_lsq, 'ci', ci)             
                
        return res_lsq, C_fit, E_fit
