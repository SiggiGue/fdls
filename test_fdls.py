"""Test suite for `fdls.py`"""
import pytest
import numpy as np

from pytest import assume
from scipy.signal import freqz, butter

from fdls import magnitude_to_minimumphase
from fdls import fdls_design


def butterworth_lowpass_magnitude(order: float, fcrit: float, samplerate: float, numtaps:int=512):
    """Returns frequency and magnitude of a butterworth filter according to given parameters.

    Args:
        order (float): Order of the butterworth filter
        fcrit (float): Edge frequency
        samplerate (float): Samplring rate (linked to fcrit)
        numtaps (int, optional): Number of frequency bins. Defaults to 512.

    Returns:
        freqs (ArrayLike): Frequency vector.
        magnitude (ArrayLike): Magnitude spectrum.
        
    """
    freqs = np.linspace(0, samplerate/2, numtaps)
    jwn = 1j * (freqs / fcrit)
    magnitude = np.sqrt(np.abs(1 / (1 + jwn**order * (-jwn)**order)))
    return freqs, magnitude


def test_fdls_design_lp_magnitude_only():
    b_des, a_des = butter(1, 0.5)
    w_des, frf_des = freqz(b_des, a_des)
    magnitude_des = np.abs(frf_des)
    b_fdls, a_fdls = fdls_design(ma_order=1, ar_order=1, magnitude=magnitude_des, phase=None)
    w_fdls, frf_fdls = freqz(b_fdls, a_fdls)
    magnitude_fdls = np.abs(frf_fdls)
    phase_fdls = np.angle(frf_fdls)
    assume(np.all(w_des == w_fdls))
    assume(np.mean((magnitude_fdls-magnitude_des)**2) < 1e-5)
    assume(np.mean((magnitude_to_minimumphase(magnitude_des) - phase_fdls)**2) < 1e-3)

def test_fdls_design_lp_magnitude_fractional_order():
    # create orde 0.5 butterworth lowpass
    freq_des, magnitude_des = butterworth_lowpass_magnitude(0.5, 10000, 44100)
    b_fdls, a_fdls = fdls_design(ma_order=3, ar_order=3, magnitude=magnitude_des, phase=None)
    w_fdls, frf_fdls = freqz(b_fdls, a_fdls, len(magnitude_des))
    magnitude_fdls = np.abs(frf_fdls)
    phase_fdls = np.angle(frf_fdls)
    assume(np.mean((magnitude_fdls-magnitude_des)**2) < 1e-5)
    assume(np.mean((magnitude_to_minimumphase(magnitude_des) - phase_fdls)**2) < 1e-3)

def test_fdls_design_lp_magnitude_phase():
    b_des, a_des = butter(2, 0.5)
    w_des, frf_des = freqz(b_des, a_des)
    magnitude_des = np.abs(frf_des)
    phase_des = np.angle(frf_des)    
    b_fdls, a_fdls = fdls_design(ma_order=2, ar_order=2, magnitude=magnitude_des, phase=phase_des)
    w_fdls, frf_fdls = freqz(b_fdls, a_fdls)
    magnitude_fdls = np.abs(frf_fdls)
    phase_fdls = np.angle(frf_fdls)
    assume(np.all(w_des == w_fdls))
    assume(np.mean((magnitude_fdls-magnitude_des)**2) < 1e-5)
    assume(np.mean((phase_des - phase_fdls)**2) < 1e-3)

def test_fdls_design_bp_magnitude_only():
    b_des, a_des = butter(2, [0.7, 0.9], 'bp')
    w_des, frf_des = freqz(b_des, a_des, 4096)
    magnitude_des = np.abs(frf_des)
    phase_des = magnitude_to_minimumphase(magnitude_des)
    # weights_des = magnitude_des/np.max(abs(magnitude_des))
    b_fdls, a_fdls = fdls_design(ma_order=4, ar_order=4, magnitude=magnitude_des, phase=phase_des)
    w_fdls, frf_fdls = freqz(b_fdls, a_fdls, 4096)
    magnitude_fdls = np.abs(frf_fdls)
    phase_fdls = np.unwrap(np.angle(frf_fdls))
    assume(np.all(w_des == w_fdls))
    assume(np.mean((magnitude_fdls-magnitude_des)**2) < 1e-4)
    assume(np.mean(((phase_des-np.mean(phase_des)) - (phase_fdls-np.mean(phase_fdls)))**2) < 1e-1)
        
if __name__ == "__main__":
    pytest.main()