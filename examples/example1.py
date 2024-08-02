"""Example 1"""
from pylab import *

from scipy.signal import freqz

import fdls

numtaps = 1024
samplerate = 44100
freq = np.linspace(0, samplerate/2, numtaps)
magnitude = 0.5*np.sin(2*np.pi*2*np.linspace(0, 1, numtaps)) - freq/samplerate + 1
plt.semilogx(freq, 20*np.log10(magnitude))

b,a = fdls.fdls_design(16, 16, magnitude, None)

freqz(b, a, 4096, fs=samplerate, plot=lambda w,h: plt.semilogx(w, 20*np.log10(np.abs(h)), '--'))

plt.legend(['desired', 'fdls design'])

show()
