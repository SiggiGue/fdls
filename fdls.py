"""Implementation of the Frequency Domain Least Squares Filter Design Method
according to Soderstrand and Berchin.

"""

import numpy as np
import numpy.typing as npt
from typing import Tuple
from scipy.signal import hilbert


def magnitude_to_minimumphase(magnitude: npt.ArrayLike, twosided: bool = False) -> np.ndarray:
    """Returns minimumphase for given onesided magnitude spectrum.

    Args:
        magnitude (ArrayLike) : Magnitude spectrum (expects onesided magnitude spectrum by default)
        twosided (bool) : Set to True if your given magnitude spectrum is twosided. Default: False

    Returns:
        phase (ndarray) : Minimum phase (onesided) for given magnitude.

    """

    # replacing zeros to prevent zero division:
    magnitude[magnitude==0] = np.finfo(magnitude.dtype).eps
    if twosided:
        magni_twosided = magnitude
    else:
        magni_twosided = np.concatenate((magnitude, magnitude[-1:0:-1]))

    phase_twosided = np.imag(-hilbert(np.log(np.abs(magni_twosided))))
    return phase_twosided if twosided else phase_twosided[:len(magnitude)]


def fdls_design(ma_order: int, 
                ar_order: int, 
                magnitude: npt.ArrayLike, 
                phase: npt.ArrayLike, 
                weights: None | npt.ArrayLike = None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns b,a filter coefficients (ARMA model) using Frequency Domain Least Squares method

    Args:
        ma_order (int) : Order of b coefficients i.e. numerator. Moving average part of the model.
        ar_order (int) : Order of a coefficients i.e. denominator. Autregressive part of the model.
        magnitude (ArrayLike[N]) : Magnitude of desired frequency response.
        phase (None or ArrayLike[N]) : Phase of desired frequency response. If None, a minimum phase is calculated from the provided magnitude.
        weights (None or ArrayLike[N]) : Frequency weighting or LS error.

    Returns:
        b_coeffs, a_coeffs (Tuple[ndarray, ndarray]) : b, a Coefficients of numerator (optimal in least squares sense).
            
    Raises:
        ValueError: If magnitude or phase are not real valued or not of the same length.
        ValueError: If magnitude, phase (if given) and weights (if given) are not of the same length.
        
    References:
        G. Berchin, "Precise Filter Design [DSP Tips & Tricks]," in IEEE Signal Processing Magazine, 
        vol. 24, no. 1, pp. 137-139, Jan. 2007, 
        DOI: 10.1109/MSP.2007.273077.
        
        Michael A. Soderstrand, Gregory Berchin & Randy S. Roberts
        (1995) Frequency domain least-squares system identification, international journal of
        electronics, 78:1, 25-35, 
        DOI: 10.1080/00207219508926137
    
    """

    if not np.all(np.isreal(magnitude)):
        raise ValueError("A real valued magnitude array is required.")    
    
    if phase is None:
        phase = magnitude_to_minimumphase(magnitude)
    else: 
        if not np.all(np.isreal(phase)):
            raise ValueError("A real valued phase array is required.")
        
    if len(magnitude) != len(phase):
        raise ValueError(f'Length of magnitude ({len(magnitude)}) and phase ({len(phase)}) must be the same.')
    
    if weights is not None and len(weights) != len(magnitude):
        raise ValueError(f'Length of weights ({len(weights)}) must be the same as length of phase and magnitude ({len(magnitude)}).')
    
    # make row vectors for delay sample orders in difference equation AR and MA part:
    kma = 1 + np.arange(ma_order)
    kar = 1 + np.arange(ar_order)

    # make column vectors for frequency dependent variables
    magnitude = np.array(magnitude).reshape(-1, 1)
    phase = np.array(phase).reshape(-1, 1)
    
    # create normalized frequency values from zero to pi
    omega = np.linspace(0, np.pi, len(magnitude)).reshape(-1, 1)       
    
    # Now build up the linear equation: y0 = Xθ​ 
    # and then solve it for Θ = (X'X)^-1 X'y0
    
    # calculate y0 (filter output at k=0), actually the y(k=0) column vector
    y0 = magnitude * np.cos(phase)
    
    # calculate the AR 'response' for frequencies and delays 
    yar = -magnitude * np.cos(-kar * omega + phase)
    
    # calculate the MA 'response' for frequencies and delays 
    uma = np.cos(-kma * omega)
    
    # design the matrix
    xmat = np.concatenate((yar, np.ones_like(omega), uma), axis=1)
    
    if weights is not None:
        # weighted least squares solution c = [X^T W^T W X]^-1 X^T W^T W y0
        wei = np.diag(np.array(weights).flatten())
        coeffs = np.squeeze(
            np.linalg.pinv(xmat.T @ wei.T @ wei @ xmat) @ xmat.T @ wei.T @ wei @ y0)
    else:
        # least squares solution: c = [X^T X]^-1 X^T y0
        coeffs = np.squeeze(
            np.linalg.pinv(xmat.T @ xmat) @ xmat.T @ y0)
        
    a_coeffs = np.concatenate(([1], coeffs[:ar_order]))
    b_coeffs = coeffs[ar_order:ar_order+ma_order+1] 
    return b_coeffs, a_coeffs
