from cmath import pi
import random
import argparse

from rich import print
from rich.live import Live
from rich.table import Table
from rich.bar import Bar
from rich.color import Color
#from gpiozero import CPUTemperature
from command_detector import CommandDetector
from threading import Thread, Lock
from pi_metrics import PiMetrics


pred_lock = Lock()
def update_predictions():
    while True:
        preds = detector.inference() * 100
        with pred_lock:
            global predictions
            predictions = preds

def get_prediction():
    with pred_lock:
        global predictions
        return predictions

def generate_table() -> Table:
    """Make a new table."""
    grid = Table()
    grid.add_column("Live CLI Inference", justify="center")    
    grid.add_row(
        f"[bold green]Model loaded:[/bold green]\r\n{FLAGS.tflite_path}"
    )

    inference_table = Table()
    inference_table.add_column("ID")
    inference_table.add_column("Word")
    inference_table.add_column("Probability")
    main = Table.grid()
    main.add_column("Info")
    main.add_column("Inference")

    grid.add_row(main)

    stats = Table(show_lines=True)
    stats.add_column("Name")
    stats.add_column("Value")
    # vol_value = random.randint(20, 30)
    # cpu_temp = pi_metrics.get_cpu_temp()
    # mem_allocation = pi_metrics.get_mem_allocation()
    # cpu_clock = pi_metrics.get_cpu_clock()
    # stats.add_row("CPU Temp.", f'{cpu_temp}°C' if cpu_temp < 60 else f'[red]{cpu_temp}°C')
    # stats.add_row("CPU Clock", f'{cpu_clock} MHz' if cpu_clock < 600 else f'[red]{cpu_clock} MHz')
    # stats.add_row("Memory used", f'{mem_allocation} MB' if mem_allocation < 400 else f'[red]{mem_allocation} MB')
    
    #stats.add_row("Volume",Bar(100,0,vol_value,width=25, color=Color.from_rgb(255,255-vol_value*2.55,255-vol_value*2.55)))
    main.add_row(
        inference_table, stats
    )
    predictions = get_prediction()
    for index, word in enumerate(WORDS):
        value = predictions[index]
        inference_table.add_row(
            f"{index}", 
            f"   {word}" if value < 50 else f"[green]-> {word}", 
            f"{value:3.2f}%" if value < 50 else f"[bold]{value:3.2f}%"
        )

    return grid

def main():
    global detector
    global predictions
    predictions = [0.0] * len(WORDS)
    detector = CommandDetector(WORDS, FLAGS.tflite_path)
    data_poller = Thread(target=update_predictions)
    data_poller.start()

    # global pi_metrics 
    # pi_metrics = PiMetrics()
    # pi_metrics.start()
    with Live(generate_table(), refresh_per_second=20) as live:
        while(True):
            live.update(generate_table())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    global WORDS
    WORDS = ["_silence_","_unknown_"] + FLAGS.wanted_words.split(",")
    main()