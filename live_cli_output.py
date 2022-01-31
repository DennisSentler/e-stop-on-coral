import random
import time
import argparse
from datetime import datetime

from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.bar import Bar
from rich.color import Color
from gpiozero import CPUTemperature

from rich import print

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
    vol_value = random.randint(20, 30)
    cpu_value = random.randint(80, 95)
    mem_value = random.randint(70, 88)

    
    stats.add_row("Volume",Bar(100,0,vol_value,width=25, color=Color.from_rgb(255,255-vol_value*2.55,255-vol_value*2.55)))
    stats.add_row("CPU",Bar(100,0,cpu_value,width=25, color=Color.from_rgb(255,255-cpu_value*2.55,255-cpu_value*2.55)))
    stats.add_row("Memory",Bar(100,0,mem_value,width=25, color=Color.from_rgb(255,255-mem_value*2.55,255-mem_value*2.55)))
    main.add_row(
        inference_table, stats
    )
    for index, word in enumerate(WORDS):
        value = random.random() * 100
        inference_table.add_row(
            f"{index}", 
            f"   {word}" if value < 50 else f"[green]-> {word}", 
            f"{value:3.2f}%" if value < 50 else f"[bold]{value:3.2f}%"
        )

    return grid

def main():
    #tflite_test(FLAGS.tflite_path, FLAGS.testdata_path)
    
    #print(f"[bold green]Model loaded:[/bold green] {FLAGS.tflite_path}")
    with Live(generate_table(), refresh_per_second=60) as live:
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
    WORDS = ["_silence_","_unknown_"] + FLAGS.wanted_words.split(",")
    main()