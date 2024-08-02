import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

from fdls import fdls_design

NUMTAPS = 4096
SAMPLERATE = 44100

# create a desired magnitude spectrum
freq = np.linspace(0, SAMPLERATE/2,  NUMTAPS)
phi = np.cumsum(np.logspace(2, 0, NUMTAPS))*2*np.pi/SAMPLERATE
magnitude = 0.5*np.cos(phi) - freq/SAMPLERATE + 1

# FDLS filter design of desired magnitude spectrum 
# with minimum phase calculated from magnitude spectrum
bcoef, acoef = fdls_design(
    ma_order=12, 
    ar_order=12, 
    magnitude=magnitude, 
    phase=None)

plt.semilogx(freq, 20*np.log10(magnitude))
freqz(
    bcoef, 
    acoef, 
    NUMTAPS, 
    fs=SAMPLERATE, 
    plot=lambda w,h: plt.semilogx(w, 20*np.log10(np.abs(h)), '--'))

plt.legend(['desired', 'fdls design'])
plt.grid(True)
plt.xlabel('Frequency in Hz')
plt.ylabel('Magnitude in dB(FS)')
plt.show()