
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation 
from scipy.io import wavfile
import sounddevice as sd
import threading
import time
from math import log
import sys


WINDOW_SIZE = 4096
STEP_SIZE = WINDOW_SIZE // 4

# helper function to check that the signal is a stereo signal
def check_stereo(signal):
    try:
        for elt in signal:
            if len(elt) != 2: return False 
    except: return False
    return True
 

def stereo_to_mono(stereo):
    """
    Converts a stereo signal into a mono signal by averaging time-domain values for L and R.

    Args:
        stereo: numpy array of numpy arrays of floats, each two elements long; represents the stereo signal
    
    Ret:
        mono: numpy array of floats; represents mono output
    """
   
    assert isinstance(stereo, np.ndarray), AssertionError("Input must be numpy array.")
    assert check_stereo(stereo), AssertionError("Input must be array of size-2 arrays.")

    # average the two sides into one signal
    return 0.5 * (stereo[:, 0] + stereo[:, 1])

def isolate_side(stereo, side):
    """
    Isolate either left or right side of a stereo signal outputted by wavfile.read
    """

    # helper function to check that the signal is a stereo signal
    def check_stereo(signal):
        try:
            for elt in signal:
                if len(elt) != 2: return False 
        except: return False
        return True
    
    assert isinstance(stereo, np.ndarray), AssertionError("Input must be numpy array.")
    assert check_stereo(stereo), AssertionError("Input must be array of size-2 arrays.")
    assert side in ["LEFT", "RIGHT"], AssertionError("Side should be either 'LEFT' or 'RIGHT'.")

    if side == "LEFT":
        return stereo[:, 0]
    else:
        return stereo[:, 1]


def stft(x, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Returns the short-time Fourier transform of signal x with a particular
        window size and step size.

    Args:
        x: list of floats, represents the sound signal
        window_size: int, the size of the analysis window when doing FFT (power of 2 for best
            results)
        step_size: int, the step/hop size by which the window moves for each STFT
    
    Ret:
        stft_arr: numpy array of numpy arrays of floats, represents the sequence of FFTs at each window
            stft_arr[i][j] is the jth DFT coefficient from the ith window
    """

    # predefine numpy array size
    size = (len(x) - window_size) // step_size + 1
    assert size > 0, AssertionError("Signal length less than window size. Reduce window size.")
    stft_arr = np.empty((size, window_size))

    # get fft for all the small windows
    for i in range(size):
        stft_arr[i,:] = np.fft.fft(x[i * step_size: i * step_size + window_size])

    return stft_arr


def k_to_hz(k, sample_rate, window_size=WINDOW_SIZE):
    """
    Convert bin number k into the corresponding frequency.

    Args:
        k: int; bin number
        sample_rate: float; sampling frequency of the signal
        window_size: int; number of samples in the window when calculating DFT

    Ret:
        freq: float; frequency corresponding to the bin k
    """
    return k * sample_rate / window_size


def hz_to_k(freq, sample_rate, window_size=WINDOW_SIZE):
    """
    Convert a frequency into the closest bin k.

    Args:
        freq: float; the frequency of interest
        sample_rate: float; sampling frequency of the signal
        window_size: int; number of samples in the window when calculating DFT

    Ret:
        k: int; the number of the bin that is closest to the frequency of interest
    """
    return round(window_size * freq / sample_rate)


def timestep_to_seconds(i, sample_rate, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Convert a time at a particular time step into real-world seconds.

    Args:
        i: int; index of the window of interest
        sample_rate: float; sampling frequency of the signal
        window_size: int; number of samples in the window when calculating DFT
        step_size: int; number of samples traversed in each step

    Ret:
        time: float; real-world time corresponding to the time step of interest
    """
    return round((i * step_size + window_size / 2) / sample_rate, 2)


def spectrogram(X):
    """
    Create the spectrogram of a signal given its STFT.

    Args:
        X: numpy array of numpy arrays of complex floats; the STFT of the signal of interest
        sample_rate
    
    Ret: 
        spgm: numpy array of numpy arrays of floats; the spectrogram where the values are the 
            magnitude squred
    """

    assert isinstance(X, np.ndarray), AssertionError("X must be a numpy array.")

    # spgm[k][i] corresponds to frequency bin k in analysis window i
    spgm = np.square(X)
    return np.transpose(spgm)


