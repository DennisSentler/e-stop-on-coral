import argparse
import pyaudio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import data as data_processor
import models

def initAudioStream():
    CHUNK = 1600
    FORMAT = pyaudio.paInt16
    SAMPLE_RATE = 16000
    INPUT_LENGTH_MS = 1000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    stream.start_stream()
    return stream


def loadModel():
    model_settings = models.prepare_model_settings(
            len(WORDS), 
            FLAGS.sample_rate, 
            FLAGS.clip_duration_ms, 
            FLAGS.window_size_ms, 
            FLAGS.window_stride_ms, 
            FLAGS.dct_coefficient_count
        )
    model = models.create_model(model_settings, FLAGS.model_architecture, FLAGS.model_size_info, False)
    model.load_weights(FLAGS.checkpoint).expect_partial()
    return model

def bytesToArray(bytes):
    return np.frombuffer(bytes, np.int16)

def saveDataPlot(figure, data, name):
    ax = figure.axes[0]
    ax.cla()
    ax.set_ylim(np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    ax.plot(data)
    figure.savefig("plots/%s.jpg" % name)
    
def preProcessAudio(audio):
    audio_scaled = np.interp(audio, (np.iinfo(np.int16).min, np.iinfo(np.int16).max), (-1, +1))
    audio_tensor = tf.convert_to_tensor(audio_scaled,  dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, [16000,1])
    audio_mfcc = data_processor.calculate_mfcc(audio_tensor, 16000, 640, 320, 10)
    audio_mfcc = tf.reshape(audio_mfcc, [-1])
    return audio_mfcc

def main():
    stream = initAudioStream()
    model = loadModel()

    # prefill
    audio = []
    audio_bytes = stream.read(FLAGS.sample_rate)
    audio.extend(bytesToArray(audio_bytes))

    #saveDataPlot(fig, audio, "testsavefig")
    #test if concatination works
    # stream.start_stream()
    # saveDataPlot(fig, audio, "0_frames_org")
    # audio_bytes = bytesToArray(stream.read(CHUNK)) 
    # saveDataPlot(fig, audio_bytes, "1_chunk")
    # tmp_frames = audio
    # tmp_frames.extend(audio_bytes)
    # saveDataPlot(fig, tmp_frames, "2_1_tmp_frames_extended")
    # del tmp_frames[:CHUNK]
    # saveDataPlot(fig, tmp_frames, "2_2_tmp_frames_del")
    # audio = tmp_frames
    # saveDataPlot(fig, tmp_frames, "3_frames_final")
    # stream.stop_stream()
    #looping concatination

    while True:
        chunk = int(FLAGS.sample_rate/10)
        audio_bytes = bytesToArray(stream.read(chunk)) 
        tmp_audio = audio
        tmp_audio.extend(audio_bytes)
        del tmp_audio[:chunk]
        audio = tmp_audio
        #saveDataPlot(fig, audio, "loop%d" % i)
        mfcc = preProcessAudio(audio)
        predictions = model(mfcc, training=False).numpy()[0]
        highes_pred = tf.argmax([predictions], axis=1).numpy()[0]
        if predictions[highes_pred] > 0.7:
            print('Prediction: %s, with %dp' % (WORDS[highes_pred], predictions[highes_pred]*100))



    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
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
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='Pretrained_models/DS_CNN/DS_CNN_L/ckpt/ds_cnn_0.95_ckpt',
        help='Checkpoint to load the weights from.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='ds_cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[6, 276, 10, 4, 2, 1, 276, 3, 3, 2, 2, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1, 276, 3, 3, 1, 1],
        help='Model dimensions - different for various models')

    FLAGS, _ = parser.parse_known_args()
    WORDS = data_processor.prepare_words_list(FLAGS.wanted_words.split(','))
    main()