import argparse
import numpy as np
import audio_provider as Audio
import interpreter
import sys
import matplotlib.pyplot as plt

def saveDataPlot(audio, mfcc, name):
    fig, (audio_axis, mfcc_axis) = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(name)
    
    #audio plot
    audio_axis.set_ylim(np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    audio_axis.plot([*range(len(audio))], audio)
    audio_axis.set_xlabel("time")
    audio_axis.set_ylabel("a")

    #mfcc plot
    mfcc_axis.set_ylabel("mfcc c.")
    mfcc_axis.set_xlabel("time")
    mfcc= np.swapaxes(mfcc[0], 0 ,1)
    mfcc_axis.imshow(mfcc, interpolation='nearest', cmap='coolwarm', origin='lower')
    plt.savefig("plot.jpg")
    plt.cla()

def main():
    mic = Audio.AudioProvider(FLAGS.input_device)
    mic.start()
    # tpu = interpreter.Intepreter(FLAGS.tflite_path, name="Test")
    # tpu.start()
    last_prediction = 11

    try:
        while True:
            #audio = mic.read_audio_window()
            audio = mic.get_mfcc()
            print(audio)
            #mfccs = Audio.pre_proc_audio(audio)
            # tpu.set_input(mfccs)
            # prediction_list = tpu.get_output()
            # if prediction_list is not None:
            #     highest_pred = np.argmax(prediction_list)
            #     if highest_pred != last_prediction:
            #         last_prediction = highest_pred
            #         if highest_pred < len(WORDS):
            #             print(f"Prediction: {WORDS[highest_pred]} with {int(prediction_list[highest_pred]*100)}%")
            #             saveDataPlot(audio, mfccs, f"{WORDS[highest_pred]}_{int(prediction_list[highest_pred]*100)}")


    except (KeyboardInterrupt, SystemExit):
        mic.stop()
        #tpu.stop()
        print("\nINFO: Applicatioon canceled!")
        sys.exit()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--input_device',
        type=int,
        default=1,
        help='Index of the prefered device from pyaudio',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=20.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='models/compiled_for_edge_tpu/ds_cnn_q_2d_edgetpu.tflite',
        help='Path to TFLite file to use for testing. Must be compiled for TPU.'
        )
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)

    FLAGS, _ = parser.parse_known_args()
    WORDS = FLAGS.wanted_words.split(',')
    main()