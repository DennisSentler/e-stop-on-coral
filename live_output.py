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

    for row in range(10):
        value = random.random() * 100
        table.add_row(
            f"{row}", f"{value:3.2f}", "[red]ERROR" if value < 50 else "[green]SUCCESS"
        )
    return table

def main():
    #tflite_test(FLAGS.tflite_path, FLAGS.testdata_path)
    print(f"[bold green]Model loaded:[/bold green] {FLAGS.tflite_path}")
    with Live(generate_table(), refresh_per_second=20) as live:
        for _ in range(4000):
            time.sleep(0.05)
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
    main()