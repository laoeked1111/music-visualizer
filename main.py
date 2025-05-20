
import analysis as an
import graphics as gr
from scipy.io import wavfile

if __name__=="__main__":

    file = input("Enter the name of the WAV file to be analyzed: ")

    try:
        fs, tune = wavfile.read(file)
        if an.check_stereo(tune):
            mode = input("Stereo signal detected. Would you like to analyze the left side (L), the right side (R), or both (B)? ")

            while mode.upper() not in "LRB":
                mode = input("Stereo signal detected. Would you like to analyze the left side (L), the right side (R), or both (B)? ")
            
            if mode.upper() == "L":
                tune = an.isolate_side(tune, "LEFT")
            elif mode.upper() == "R":
                tune = an.isolate_side(tune, "RIGHT")
            else:
                tune = an.stereo_to_mono(tune)

        tune_stft = an.stft(tune)
        gr.animate_and_play(tune, fs, tune_stft, interval= 8)

        disp_sgram = input("Display spectrogram (Y/N): ")

        if(disp_sgram.upper() == "Y"):
            gr.plot_spectrogram(an.spectrogram(tune_stft), fs)


    except:
        print("Something failed. Please make sure that your file name is of the form \"file.wav.\"")