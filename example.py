## EXAMPLE SCRIPT FOR NEST.PY
##
## Bedartha Goswami, November 4, 2013

import numpy as np
import nest
import matplotlib.pyplot as plt

# initialize the data
nT, nL = 2000, 350              # lengths of signal and lags
t = np.linspace(0, 300, nT)     # time axis
T1, T2 = 20., 10.               # periods of sine components
w1, w2 = (2 * np.pi) / T1, (2 * np.pi) / T2 # frequencies
x = np.sin(w1 * t) + np.sin(w2 * t)         # signal
lags = np.arange(-nL/2, nL/2 + 1)           # lag vector

print("testing for:\n\t length of data :: %d\n\t length of lag  :: %d"\
           %(len(x), len(lags)))
acf = nest.similarity(t, x, t, x, lag=lags, method='gXCF')
power, freq = nest.pspek(acf, np.mean(np.diff(lags, axis=0)))

# plot the results
fig = plt.figure(figsize=[10,8])
## subplot 1 :: the signal
plt.subplot(311, position=[0.1, 0.7, 0.8, 0.225])
plt.plot(t, x, 'k')
plt.xlabel("Time")
plt.ylabel("Signal")
## subplot 2 :: the ACF
plt.subplot(312, position=[0.1, 0.4, 0.8, 0.225])
plt.plot(lags, acf, 'k-')
plt.xlabel("Lags")
plt.ylabel("ACF")
## subplot 3 :: the power spectrum
plt.subplot(313, position=[0.1, 0.1, 0.8, 0.225])
plt.plot(1. / freq, power, 'k-')
plt.xlabel("Time Period")
plt.ylabel("Power")
plt.show()
