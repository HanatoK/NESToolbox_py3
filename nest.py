"""Barebones translation of core functions of NESTOOLBOX.

NESTOOLBOX -- TOOLBOX FOR THE ANALYSIS OF NON-EQUIDISTANTLY
SAMPLED TIME SERIES is a software package developed for
Matlab and Octave by Kira Rehfeld (rehfeld@pik-potsdam.de).

More information at: http://tocsy.pik-potsdam.de/nest.php

REQUIREMENTS
------------
	:: Python 2.7 (or above)
	:: Numeric Python (NumPy)
	:: Scientific Python (Scipy)
	:: C++/C compiler (for scipy.weave.inline)
	:: Matplotlib (for visulaization of results)
	   (Matplotlib is optional but strongly recommended)

TESTED WITH
-----------
	:: Python 2.7.5
	:: NumPy 1.7.1
	:: SciPy 0.13.0

To see a list of the functions contained in this module type:
	>>> import nest
	>>> help(nest)

To see the help doc for a particular function called 'foo':
	>>> import nest
	>>> help(nest.foo)

Translated from Matlab to Python by:
Bedartha Goswami (goswami@pik-potsdam.de)

Created: October 30, 2013
Last revised: November 4, 2013
"""
import sys
import numpy as np
from numba import jit
from progressbar import ProgressBar
#from scipy.weave import inline
from scipy.fftpack import fft, fftfreq

def _chk_input(tx, x, ty, y, lag, verbose):
	"""Checks input for consistency and parse optional parameters.

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	nx, ny = len(x), len(y)
	if nx != len(tx) or ny != len(ty):
		err_msg = "Vectors for sampling times and observations " \
			  "have to be of same length."
		sys.exit(err_msg )
	tmin, tmax = max(min(tx), min(ty)), min(max(tx), max(ty))
	if tmax - tmin < np.spacing(1):
		sys.exit("The time-series do not overlap.")
	if type(lag) == type(None):
		T = 0.5 * (tmax - tmin)
		dT = 0.1 * T
		lag = np.arange(-T, T, dT)
	if nx == ny and not any(x - y):
		flag_acf, showType = 1, "Calculate ACF"
	else:
		flag_acf, showType = 0, "Calculate XCF"
	if verbose: print(showType)
	return lag, flag_acf
	
def _norm_data(tx, x, ty, y, lag):
	"""Normalizes the data, i.e., reduces them to dimensionless variables.

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	# make time axes dimensionless
	if len(lag) == 1:
		dtlag = 1.
	else:
		xx = np.mean(np.diff(tx, axis=0))
		yy = np.mean(np.diff(ty, axis=0))
		dtlag = max(xx, yy)
	tx, ty, normlag = tx / dtlag, ty / dtlag, lag / dtlag
	# normalize data to standard Z-score
	x, y = normalize(x), normalize(y)
	return tx, x, ty, y, normlag

def _get_threshold(sigma, tau=10.):
	"""Clacluates threshold beyond which Gaussian is zero.

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	return float(np.sqrt(2 * (sigma ** 2)  * tau))

def _chk_lag_inside(lag, mindist, maxdist):
	"""Checks if a given lag falls within (mindist, maxdist)

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	return (lag <= mindist) + (lag >= maxdist)

def _chk_too_few_pts(lag, tdist, sigma):
	"""Checks if Gaussian kernel has less than 5 data points.

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	thr = _get_threshold(sigma)
	tdist, ii = _use_numba(tdist, np.array(lag), thr)
	s = tdist[ii]
	kern_content = (s <= 5 * sigma).sum() + (s >= -5 * sigma).sum()
	return kern_content < 5, s, ii

def _get_warning(cond1, cond2):
	"""Obtains the warning messages for the given pair of conditions.

	This function returns the warning messages corresponding to the
	boolean conditions that are obtained using `_chk_lag_inside` and
	`_chk_too_few_pts`.

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	header = "Warning: Correlation array will contain NANs!"
	warn = {'cond1': "Lag is outside time difference range",
		'cond2': "Number of obs. points inside kernel is less than 5"
			+ "\nConsider smaller range for lags."
		}
	msg = header + "\n"
	msg = msg + warn['cond1'] * cond1 + "\n" +  warn['cond2'] * cond2
	return msg

def _norm_corr(tx, x, ty, y, sigma, C):
	"""Normalizes the estimated correlation array to [-1, 1].

	Note
	----
	This is an internal function. Please be careful in changing it.
	"""
	tdistx = not_dist(tx, tx, func='diff').flatten()
	tdisty = not_dist(ty, ty, func='diff').flatten()
	thr = _get_threshold(sigma)
	selectx = np.where(np.abs(tdistx) <= thr)[0]
	selecty = np.where(np.abs(tdisty) <= thr)[0]
	buffx, buffy = tdistx[selectx], tdisty[selecty]
	WLx, WLy = gauss(buffx, sigma), gauss(buffy, sigma)
	weightx, weighty = WLx.sum(), WLy.sum()
	xdist, ydist = not_dist(x, x, func='prod'), not_dist(y, y, func='prod')
	xdist, ydist = xdist.flatten(), ydist.flatten()
	# corrected variance as weighted mean
	varX = (xdist[selectx] * WLx).sum() / weightx
	varY = (ydist[selecty] * WLy).sum() / weighty
	# normalize the correlation entries
	C = C / (np.sqrt(varX) * np.sqrt(varY))
	if any(C.flatten() > 1.):
		C = C / C.max()
	return C	

#def _use_weave(tdist, lag, thr):
#	"""Uses `scipy.weave.inline` to get distances over threshold.
#
#	Returns a smaller distance array which contains only those
#	distance values for which an evaluated Gaussian is non-zero.
#	The threshold is determined be `_get_threshold`for a given
#	kernel width `sigma`. 
#	This greatly minimizes the computation  time because:
#	(i) most of the distances are larger than the threshold, and
#	(ii) using C++ greatly speeds up computation
#
#	Note
#	----
#	This is an internal function. Please be careful in changing it.
#	"""
#	l = len(tdist)
#	tdist_d, select = np.empty(tdist.shape), []
#	code = """
#		double arr_contents;
#		double test;
#                arr_contents = *tdist;
#                for (int n=0; n<l; n++){
#                	tdist_d[n] = arr_contents + *lag;
#			test = fabs (tdist_d[n]);
#			if (test<thr) {
#					select.append (n); // py::list
#				}
#                        tdist++;
#                        arr_contents = *tdist;
#                }
#	"""
#	help_code = "#include <math.h>       /* fabs */"
#	inline(code,
#		   ['tdist_d', 'select', 'tdist', 'lag', 'thr', 'l'],
#		   support_code=help_code)
#	return tdist_d, select

@jit
def _use_numba(tdist, lag, thr):
    l = len(tdist)
    tdist_d, select = np.empty(np.shape(tdist)), []
    for n in range(l):
        tdist_d[n] = tdist[n] + lag
        test = np.abs(tdist_d[n])
        if (test < thr):
            select.append(n)
    return tdist_d, select



def gauss(t, sigma):
	"""Computes the Gaussian function.

	Returns the value of the Gaussian function for given input.

	Parameters
	----------
	t	: array_like
		  Array with one or more entries at which Gaussian
		  is to be evaluated.
	sigma	: float, scalar
		  Standard deviation of the Gaussian function.

	Returns
	-------
	Y	: array_like
		  The value(s) of the Gaussian function evaluated at
		  the values of the entries of `t`. `Y` has the same
		  dimensions as `t`.
	"""
	h2 = 2 * sigma ** 2
	h2pi = np.sqrt(h2 * np.pi)
	return (1 / h2pi) * (np.exp(-t ** 2 / h2))

def similarity(tx, x, ty, y,
	       lag=None, method='gXCF', SIGMAxcf=0.25,
	       verbose=True):
	"""Compute similarity coefficient between two time series.

	Returns the coefficient of similarity between two irregularly
	time series according to the theory described in:
	Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J.
	Comparison of correlation analysis techniques for irregularly
	sampled time series
	Nonlin. Proc. Geophys., 18(3), 389-404, 2011.

	Parameters
	----------
	tx, ty	: array_like (1-D)
		  Arrays containing the times of observations of two
		  processes respectively. The are not necessarily
		  co-eval, i.e., they can be irregularly sampled.
	x, y	: array_like (1-D)
		  Arrays containing the observations of processes X
		  and Y made at the corresponding entries of tx and ty
		  respectively.
	lag	: array_like (1-D, dtype=int, or None), optional
		  Integer array denoting the lags at which the similarity
		  estimation is to be carried out. If `lag`is not 
		  specified, i.e., `lag = None`, then a default lag is
		  chosen as:
		  `tmin = max(min(tx), min(ty))`
		  `tmax = min(max(tx), max(ty))`
		  `T = 0.5 * (tmax - tmin)`
		  `dT = 0.1 * T`
		  `lag = np.arange(-T, T, dT)`
	method	: string
		  String specifying the type of statistic to be used to
		  estimate the coeffficient of similarity.
		  'gXCF'	Cross-correlation, Gaussian kernel
		  **NOT IMPLEMENTED YET** {
		  'iXCF'	Cross-correlation, interpolated
		  'gMI'		Mutual Information, Gaussian Kernel
		  'iMI'		Mutual Information, interpolated
		  'ES'		Event Synchronization
		  'all'		All of the above, gives *Link Strength*
		  }
	sigmaXCF: float, scalar
		  Size of the Gaussian kernel, when applicable.
		  Default is `0.25`.
	verbose	: boolean
		  Set to `True` if for informative execution and `False`
		  for silent execution of the function. Default is `True`.

	Returns
	-------
	S	: array (1-D)
		  If `lag = None` then `S`contains only one value, i.e., 
		  the similarity coeeficient at lag zero. Otherwise, it
		  is a 1-D array with a length equal to that  of `lag`.

	Notes
	-----
	Right now, only the `gXCF`method has been implemented. Thus,
	`nest.similarity` is the same as `nest.nexcf` as of now. This 
	is because `nest.similarity` simply preprocesses the various
	options for simliarity estimation and passes the main arguments
	to the `nexcf`function.

	Example
	-------
	>>> nTx, nTy = 100, 200 # no. of observations of X and Y
	>>> tx, ty = np.linspace(0., 100., nTx), np.linspace(0., 100., nTy)
	>>> wx, wy = (2 * np.pi) / 10., (2 * np.pi) / 20. # frequencies
	>>> x, y = np.sin(wx * tx),  np.sin(wy * ty) # observations
	>>> lags = np.arange(-50., 50.) # lags at which to get similarity
	>>> S = nest.similarity(tx, x, ty, y, lag=lags, method='gXCF')
	>>> # if you want to plot the results then...
	>>> import matplotlib.pyplot as plt
	>>> plt.plot(lags, S, 'ko-')
	>>> plt.xlabel("Lags (time units)")
	>>> plt.ylabel("Coefficient of Similarity")
	>>> plt.ylim(-0.12, 0.12)
	>>> plt.show()
	"""
	if method not in ["gXCF", "GXCF", "gxcf"]:
		err_msg = "METHOD error: Only Gaussian kernel method" \
			  "implemented till now! Please use method='gXCF'"
		sys.exit(err_msg)
	return nexcf(tx, x, ty, y, lag, SIGMAxcf, verbose)

def nexcf(tx, x, ty, y, lag=None, sigma=0.25, verbose=True):
	"""Computes correlation between two irregularly sampled time series.

	Returns the correlation coefficient  between two irregularly
        time series according to the theory described in:
        Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J.
        Comparison of correlation analysis techniques for irregularly
        sampled time series
        Nonlin. Proc. Geophys., 18(3), 389-404, 2011.

        Parameters
        ----------
        tx, ty  : array_like (1-D)
                  Arrays containing the times of observations of two
                  processes respectively. The are not necessarily
                  co-eval, i.e., they can be irregularly sampled.
        x, y    : array_like (1-D)
                  Arrays containing the observations of processes X
                  and Y made at the corresponding entries of tx and ty
                  respectively.
        lag     : array_like (1-D, dtype=int, or None), optional
                  Integer array denoting the lags at which the similarity
                  estimation is to be carried out. If `lag`is not 
                  specified, i.e., `lag = None`, then a default lag is
                  chosen as:
                  `tmin = max(min(tx), min(ty))`
                  `tmax = min(max(tx), max(ty))`
                  `T = 0.5 * (tmax - tmin)`
                  `dT = 0.1 * T`
                  `lag = np.arange(-T, T, dT)`
	sigma	: float, scalar
                  Size of the Gaussian kernel, when applicable.
                  Default is `0.25`.
        verbose : boolean
                  Set to `True` if for informative execution and `False`
                  for silent execution of the function. Default is `True`.

        Returns
        -------
        C       : array (1-D)
                  If `lag = None` then `S`contains only one value, i.e., 
                  the similarity coeeficient at lag zero. Otherwise, it
                  is a 1-D array with a length equal to that  of `lag`.

        Example
        -------
	>>> nTx, nTy = 100, 200 # no. of observations of X and Y
        >>> tx, ty = np.linspace(0., 100., nTx), np.linspace(0., 100., nTy)
        >>> wx, wy = (2 * np.pi) / 10., (2 * np.pi) / 20. # frequencies
        >>> x, y = np.sin(wx * tx),  np.sin(wy * ty) # observations
        >>> lags = np.arange(-50., 50.) # lags at which to get similarity
        >>> S = nest.nexcf(tx, x, ty, y, lag=lags)
        >>> # if you want to plot the results then...
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(lags, S, 'ko-')
        >>> plt.xlabel("Lags (time units)")
        >>> plt.ylabel("Coefficient of Similarity")
        >>> plt.ylim(-0.12, 0.12)
        >>> plt.show()
	"""
	lag, flag_acf = _chk_input(tx, x, ty, y, lag, verbose)
	tx, x, ty, y, nlag = _norm_data(tx, x, ty, y, lag)
	C = np.zeros(len(lag))
	tdist = not_dist(tx, ty, func='diff').flatten()
	xydist = not_dist(x, y, func='prod').flatten()
	mindist, maxdist = tdist.min(), tdist.max()
	if verbose: pbar = ProgressBar(maxval=len(lag)).start()
	for i in range(len(lag)):
		c1 = _chk_lag_inside(nlag[i], mindist, maxdist)
		c2, tdD, ii = _chk_too_few_pts(nlag[i], tdist, sigma)
		if c1 or c2:	## If either condition is TRUE...
			C[i] = np.nan
			msg = _get_warning(cond1, cond2)
			print("Warning! At lag %d"%nlag[i])
			print(msg)
		else:		## calculate correlation
			WL = gauss(tdD, sigma)
			weight = WL.sum()
			C[i] = (xydist[ii] * WL).sum() / weight
		if verbose: pbar.update(i + 1)
	if verbose: pbar.finish()
	C = _norm_corr(tx, x, ty, y, sigma, C)
	return C

def normalize(arr, axis=None):
	"""Normalizes given array to the standard Z-score.

	Returns the standard normal score of the array elements. The
	score is based on the mean and standard deviation of the array
	elements. 
	
	Parameters
	----------
	arr	: array_like (1D)
		  Array whose standard score is desired. `arr`is 
		  passed on to `numpy.mean` and `numpy.std`, so if
		  `arr`is not an array, a conversion is attempted.
	axis	: TO BE IMPLEMENTED

	Returns
	-------
	z	: 1-D numpy array with same dtype as input array.
	"""
	return (arr - arr.mean()) / arr.std()

def not_dist(x, y, func='diff'):
	"""Compute a (not) 'distance' array based on difference/product.

	Returns pairwise 'distances' between all entries of a pair of
	1-D arrays. The distances are based on either the difference
	of the entries or their product.

	Parameters
	----------
	x, y	: array_like
		  1-D arrays contatining entries whose 'distances' are
		  to be computed -- not necesarily of the same length.
	func	: string specifying which function to use to compute the
		  'distance'.
		  'diff'	difference between entries x[i] and y[j]
		  'prod'	product of the entries x[i] and y[j]

	Returns
	-------
	ndist	: 2-D array of dimensions N x M, where N and M are the
		  lengths of the input arrays `x` and `y` respectively.

	Example
	-------
	>>> x = np.random.rand(100)
	>>> y = np.random.rand(50)
	>>> ndist = nest.not_dist(x, y, func='diff')
	>>> print ndist.shape
	(100, 50)

	Notes
	-----
	I feel that this is not really a distance as the values can be
	negative. I simply think of it as evaluation of a specified 
	function between all possible combinations of entries of two 
	given 1-D arrays.
	"""
	ndist = np.zeros((len(x), len(y)))
	for i in range(len(x)):
		if func == "diff":
			ndist[i] = (x[i] - y).squeeze()
		elif func == "prod":
			ndist[i] = (x[i] * y).squeeze()
	return ndist

def nearest_power_of_two(N):
	"""Computes the nearest power of 2 for a given integers N.

	Returns non-negative integer P such that P is the expression
	2^P - N is minimum.

	Parameters
	----------
	N 	: integer
		  The integer for which the nearest power of 2 is desired.

	Returns
	-------
	P	: integer
		  An integer for which the expression 2^P - N is minimum.

	Examples
	--------
	>>> N = 1011
	>>> P = nest.nearest_power_of_two(N)
	>>> P
	1024
	>>> N = 1924
	>>> P = nest.nearest_power_of_two(N)
	>>> P
	2048
	"""
	i = range(31) # for index > 30, scipy gives negative results
	poweroftwo = np.power(2, i)
	idx = np.argmin(abs(N - poweroftwo))
	return poweroftwo[idx]

def pspek(acf, dt, verbose=True):
	"""Compute power spectrum using the autocorrelation function (ACF).

	Returns the absolute value of the Fourier Transform of a given
	autocorrelation function. Then, by the Wiener-Khinchin theorem
	the returned value is equivalent to the power spectral density
	of the signal from which the ACF  was obtained. This provides 
	a handy way to estimate teh spectral density of time series with
	irregular time sampling 

	Parameters
	----------
	acf	: array_like (1-D)
		  The autocorrelation function of the signal for which
		  the power spectrum is to be obtained. This can be
		  obtained using `nest.similarity` or `nest.nexcf`.
	dt	: array_like (1-D)
		  Spacing of the ACF lags. This can be simply obtained
		  by `dt = np.mean(np.diff(lags))`
	verbose : boolean
                  Set to `True` if for informative execution and `False`
                  for silent execution of the function. Default is `True`.

	Returns
	-------
	power	: array_like (1-D)
		  The power of the signal from the input ACF was obtained.
	freq	: array_like (1-D)
		  The frequencies at which the power is estimated.

	Example
	-------
	>>> nT, nL = 2000, 100 # length of signal and lag vector
	>>> t = np.linspace(0, 300, nT)
	>>> T1, T2 = 20., 10. # periods of sine components in signal
	>>> w1, w2 = (2 * np.pi) / T1, (2 * np.pi) / T2 # frequencies
	>>> x = np.sin(w1 * t) + np.sin(w2 * t) # signal
	>>> lags = np.arange(-nL/2, nL/2 + 1)
	>>> acf = nest.similarity(t, x, t, x, lag=lags, method='gXCF')
	>>> power, freq = nest.pspek(acf, np.mean(np.diff(lags, axis=0)))
	>>> # if you want to plot the spectrum...
	>>> fig = plt.figure(figsize=[10,8])
	>>> plt.subplot(311, position=[0.1, 0.7, 0.8, 0.225])
	>>> plt.plot(t, x, 'k')
	>>> plt.xlabel("Time")
	>>> plt.ylabel("Signal")
	>>> plt.subplot(312, position=[0.1, 0.4, 0.8, 0.225])
	>>> plt.plot(lags, acf, 'k-')
	>>> plt.xlabel("Lags")
	>>> plt.ylabel("ACF")
	>>> plt.subplot(313, position=[0.1, 0.1, 0.8, 0.225])
	>>> plt.plot(1. / freq, power, 'k-')
	>>> plt.xlabel("Time Period")
	>>> plt.ylabel("Power")
	>>> plt.show()
	"""
	NFFT = nearest_power_of_two(len(acf))
	if verbose:
		msg = "Computing FFT of ACF with %d frequencies..."%NFFT
		print(msg)
	Fs = 1/dt; #  Sampling frequency
	T = 1/Fs;  #  Sampling period
	N = len(acf)
	FFT = fft(acf, n=NFFT) / len(acf)
	freq = fftfreq(NFFT, dt)
	power = abs(FFT)
	idx = freq > 0.
	power, freq = power[idx], freq[idx]
	return power, freq

