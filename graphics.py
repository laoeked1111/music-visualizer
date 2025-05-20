
from analysis import *

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



