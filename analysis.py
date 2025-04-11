
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


def stereo_to_mono(stereo):
    """
    Converts a stereo signal into a mono signal by averaging time-domain values for L and R.

    Args:
        stereo: numpy array of numpy arrays of floats, each two elements long; represents the stereo signal
    
    Ret:
        mono: numpy array of floats; represents mono output
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


def plot_spectrogram(sgram, sample_rate, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    Plot the spectrogram of the signal.

    Args:
        sgram: numpy array of numpy arrays; the spectrogram
        sample_rate: float; sampling frequency of the signal
        window_size: int; number of samples in the window of the DFT
        step_size: int; number of samples by which the window moves for the STFT

    Ret:
        None 
    """

    width = len(sgram[0])
    height = len(sgram)//2+1  # only plot values up to N/2

    plt.imshow([[log(i + sys.float_info.min) for i in j] for j in sgram[:height+1]], aspect=width/height)
    plt.axis([0, width-1, 0, height-1])

    ticks = ticker.FuncFormatter(lambda x, pos: '{0:.1f}'.format(timestep_to_seconds(x, sample_rate)))
    plt.gca().xaxis.set_major_formatter(ticks)
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:.0f}'.format(k_to_hz(y, sample_rate)))
    plt.gca().yaxis.set_major_formatter(ticks)

    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    plt.title("Spectrogram")

    plt.colorbar()
    plt.show()


def animate_and_play(tune, fs, stft_arr, interval):
    """
    Play an audio signal and simultaneously display the frequency components at each moment in time.

    Args:
        tune: 1D numpy array of floats; represents the sound signal
        fs: float; represents the sampling frequency
        stft_arr: numpy array of arrays of floats; represents the STFT of the sound signal
        interval: int; the interval at which samples of the STFT are displayed

    Ret:
        None
    """

    total_frames = stft_arr.shape[0]
    ks = interval * np.arange(WINDOW_SIZE // (2 * interval) + 1)
    y = np.zeros(len(ks))
    sampled_stft_arr = stft_arr[:, ::interval]

    fig, ax = plt.subplots()
    ax.set_xlim(0, len(ks))
    ax.set_ylim(0, np.max(np.abs(sampled_stft_arr)))

    stem_container = ax.stem(ks, y)
    ax.set_xlabel("k")
    ax.set_title("Frequency Spectrum")
    markerline, stemlines, _ = stem_container.markerline, stem_container.stemlines, stem_container.baseline

    start_time = time.perf_counter()

    # function for updating frames of the animation
    def update(_):
        elapsed_time = time.perf_counter() - start_time 
        curr_frame = int(elapsed_time * fs / STEP_SIZE)

        if curr_frame >= total_frames:
            curr_frame = total_frames - 1

        y = np.abs(sampled_stft_arr[curr_frame, 0:len(ks)])
        markerline.set_ydata(y)

        segments = [((x_, 0), (x_, y_)) for x_, y_ in zip(ks, y)]
        stemlines.set_segments(segments)

        return markerline, stemlines 

    ani = FuncAnimation(fig, update, frames= len(stft_arr), interval= 10, repeat= False)

    # play audio
    def play_audio():
        sd.play(tune, fs)
        sd.wait()

    audio_thread = threading.Thread(target= play_audio)
    audio_thread.start()

    plt.show()



if __name__=="__main__":

    # fs, tune = wavfile.read('mystery4.wav')


    # mono_signal = stereo_to_mono(tune)
    # animate_and_play(tune, fs, stft(tune), interval= 8)
    # mono_analysis(fs, [tune[i][0] for i in range(len(tune))])
    # make_animation([0 for _ in range(100)])
    # print(stft(tune)[0])



    # print(tune)
    # print(stereo_to_mono(tune))

    # my_stft = stft(tune)
    # sgram = spectrogram(my_stft)
    # plot_spectrogram(sgram, fs)

    fs, tune = wavfile.read("Electric Heartbeat.wav")
    tune = isolate_side(tune, "LEFT")
    animate_and_play(tune, fs, stft(tune), interval= 8)
    my_stft = stft(tune)
    sgram = spectrogram(my_stft)
    plot_spectrogram(sgram, fs)