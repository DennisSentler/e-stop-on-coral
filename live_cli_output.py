import random
import time
import argparse

from rich.live import Live
from rich.table import Table
from rich import print


def generate_table() -> Table:
    """Make a new table."""
    table = Table()
    table.add_column("ID")
    table.add_column("Word")
    table.add_column("Probability")

    for index, word in enumerate(WORDS):
        value = random.random() * 100
        table.add_row(
            f"{index}", 
            f"   {word}" if value < 50 else f"[green]-> {word}", 
            f"{value:3.2f}%" if value < 50 else f"[bold]{value:3.2f}%"
        )
    return table

def main():
    #tflite_test(FLAGS.tflite_path, FLAGS.testdata_path)
    main_table = Table(title="Live CLI Inference")
    main_table.add_column(f"[bold green]Model loaded:[/bold green] {FLAGS.tflite_path}", justify="center", no_wrap=True)
    #print(f"[bold green]Model loaded:[/bold green] {FLAGS.tflite_path}")
    with Live(generate_table(), refresh_per_second=5) as live:
        while(True):
            time.sleep(0.2)
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