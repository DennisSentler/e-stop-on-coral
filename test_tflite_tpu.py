# Copyright © 2021 Arm Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to run inference and test keyword spotting models in tflite format."""

import argparse
import tflite_runtime.interpreter as tflite
import numpy as np
import sklearn.metrics
import glob
import re
import measure

# function for collecting data set from file system
def load_testdata_mfcc(path):
    files = glob.glob(path + "/label_*_nr_*.npy")
    regex = re.compile("label_([\d]+)_nr")
    test_data = []
    for file in files:
        # read label number from filename
        label = int(regex.search(file).group(1))
        test_data += [(np.load(file), label)]
    return test_data

def load_interpreter(tflite_path):
    interpreter = tflite.Interpreter(
                tflite_path, 
                experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
        )
    interpreter.allocate_tensors()
    return interpreter

def tflite_test(tflite_path, testdata_path):
    """Calculate accuracy and confusion matrices on the test set.

    A TFLite model used for doing testing.

    Args:
        tflite_path: Path to TFLite file to use for inference.
    """
    interpreter = load_interpreter(tflite_path)
    test_data = load_testdata_mfcc(testdata_path)
    clock = measure.InferenceClock()

    expected_indices = [y for x, y in test_data]
    predicted_indices = []

    print("Running testing on test set...")
    mfcc_counter = 0
    for mfcc, label in test_data:
        clock.start()
        prediction = tflite_inference(mfcc, interpreter)
        clock.stop()
        predicted_indices.append(np.argmax(prediction))
        mfcc_counter += 1

    test_accuracy = sklearn.metrics.accuracy_score(predicted_indices, expected_indices)
    confusion_matrix = sklearn.metrics.confusion_matrix(expected_indices, predicted_indices)
    print(confusion_matrix)

    print(clock.report())
    print(f'Test accuracy = {test_accuracy * 100:.2f}%'
          f', (N={mfcc_counter})')

def tflite_inference(input_data, interpreter):
    """Call forwards pass of TFLite file and returns the result.

    Args:
        input_data: Input data to use on forward pass.
        interpreter: Intepreter from loaded tflite model.

    Returns:
        Output from inference.
    """

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype == np.int8:
        input_scale, input_zero_point = input_details[0]["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    input_data = input_data / input_scale + input_zero_point
    input_data = np.round(input_data) if input_dtype == np.int8 else input_data

    if output_dtype == np.int8:
        output_scale, output_zero_point = output_details[0]["quantization"]
    else:
        output_scale, output_zero_point = 1, 0


    interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_dtype))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

    return output_data

def main():
    tflite_test(FLAGS.tflite_path, FLAGS.testdata_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--testdata_path',
        type=str,
        default='testdata_as_mfcc',
        help="""\
        Folder path where to load the MFCC dataset from. 
        Data format: npy. Shape: (1,49,10), Filenames: label_#_nr_#.npy
        """)
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
    main()
