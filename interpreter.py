from operator import contains
import tflite_runtime.interpreter as tflite
import numpy as np
import threading

class Intepreter(threading.Thread):
    def __init__(
            self, 
            tflite_path: str, 
            name="Custom Model", 
        ):
        """
        This class is only for tflite on TPU compiled models.
        Implements Thread for running inference continuously. 

        Args:
            tflite_path (str): Path to compiled and converted tflite file
            name (str, optional): Defaults to "Custom Model".
        """
        threading.Thread.__init__(self)
        self.daemon=True
        self._input_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._input_data = [None]
        self._output_data = [None]
        self._running = False
        self._tflite_intepreter = tflite.Interpreter(
                    tflite_path, 
                    experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
            )
        self._tflite_intepreter.allocate_tensors()

    def set_input(self, input_data):
        """
        Sets the input data for the interpreter to inference on.

        Args:
            input_data (ndarray): please check the input data shape and align on that
        """
        with self._input_lock:
            self._input_data = input_data

    def get_output(self):
        """
        Returns the latest output of the inferencing interpreter, with probabilities.
        Make sure, the interpreter is running and receiving input data.

        Returns:
            ndarray: 1D array of data with 'float32' type.
            May return '[None]'.
        """
        with self._output_lock:
            return self._output_data[0]

    def run(self):
        """
        Run function implemented for Thread.start()
        """
        self._running = True
        while self._running:
            with self._input_lock:
                current_input = self._input_data
            
            if None not in current_input:
                current_output = self._inference(current_input)
                with self._output_lock:
                    self._output_data = current_output

    def stop(self):
        self._running = False

    def get_input_details(self):
        """
        Returns the input details of the currently loaded model/interpreter.

        Returns:
            list[dict[str, Any]]: A list of input details.
        """
        return self._tflite_intepreter.get_input_details()

    def _inference(self, input_data):
        """
        Copied from 
        https://github.com/ARM-software/ML-examples/blob/master/tflu-kws-cortex-m/Training/test_tflite.py
        """
        input_details = self._tflite_intepreter.get_input_details()
        output_details = self._tflite_intepreter.get_output_details()

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


        self._tflite_intepreter.set_tensor(input_details[0]['index'], input_data.astype(input_dtype))
        self._tflite_intepreter.invoke()

        output_data = self._tflite_intepreter.get_tensor(output_details[0]['index'])

        output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

        return output_data