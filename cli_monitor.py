import argparse
import sys
from rich import print
from rich.live import Live
from rich.table import Table
from command_detector import CommandDetector, CLOCK_INFERENCE, CLOCK_PRE_PROC, CLOCK_TOTAL
from threading import Thread, Lock
from measure import clock
from config import MODEL
from logger import log

CLOCKS = [CLOCK_PRE_PROC, CLOCK_INFERENCE, CLOCK_TOTAL]

pred_lock = Lock()
def update_predictions():
    while True:
        preds = detector.inference() * 100
        with pred_lock:
            global __predictions
            __predictions = preds

def get_prediction():
    with pred_lock:
        global __predictions
        return __predictions

def generate_table() -> Table:
    """Make a new table."""
    main_table = Table()
    main_table.add_column("Live CLI Inference", justify="center")    
    main_table.add_row(
        f"[bold green]Model loaded:[/bold green]\r\n{FLAGS.tflite_path}"
    )

    sub_table = Table.grid()
    sub_table.add_column()
    infrence_table = Table(expand = True)
    clocks_table = Table(expand = True)

    infrence_table.add_column("ID")
    infrence_table.add_column("Word")
    infrence_table.add_column("Probability")

    predictions = get_prediction()
    for index, word in enumerate(MODEL['words']):
        value = predictions[index]
        infrence_table.add_row(
            f"{index}", 
            f"   {word}" if value < 50 else f"[green]-> {word}", 
            f"{value:3.2f} %" if value < 50 else f"[bold]{value:3.2f} %"
        )

    clocks_table.add_column("Name")
    clocks_table.add_column("OP/s", justify="right")
    clocks_table.add_column("Std. diviation", justify="right")
    clocks_table.add_column("Min. time", justify="right")
    clocks_table.add_column("Max. time", justify="right")

    for c_name in CLOCKS:
        ops = clock.calculate_avg_OPS(c_name)
        std_div = clock.get_std_deviation(c_name)
        min, max = clock.get_min_max_values(c_name)
        clocks_table.add_row(
            f"{c_name.lower()}", 
            "-" if ops is None else     f"{ops:.2f}",
            "-" if std_div is None else f"{std_div:.2f}",
            "-" if min is None else     f"{min} ms",
            "-" if max is None else     f"{max} ms"
        )
    sub_table.add_row(infrence_table, clocks_table)
    main_table.add_row(sub_table)
    return main_table

def main():
    global detector
    global __predictions
    __predictions = [0.0] * len(MODEL['words'])
    detector = CommandDetector(FLAGS.tflite_path)
    data_poller = Thread(target=update_predictions, daemon=True)
    data_poller.start()
    log.info("Start CPLI monitor")
    try:
        with Live(generate_table(), refresh_per_second=FLAGS.cli_refresh_per_s) as live:
            while(True):
                live.update(generate_table())
    except KeyboardInterrupt:
        log.info("Application shutdown")
        sys.exit()
    except Exception as e:
        log.exception(e)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='models/compiled_for_edge_tpu/ds_cnn_q_2d_edgetpu.tflite',
        help='Path to TFLite file to use for testing. Must be compiled for TPU.'
        )
    parser.add_argument(
        '--cli_refresh_per_s',
        type=int,
        default=20,
        help='Sets the amount of refreshes per second for the cli interface.'
        )
    FLAGS, _ = parser.parse_known_args()
    main()