import time
import numpy as np
from threading import Lock

__CAPACITY = 2000
__clocks = {}
__lock = Lock()

def addClock(name):
    with __lock:
        if name in __clocks:
            raise Exception(f'"{name}" already exists in clock list')

        __clocks[name] = {"times_ms":[], "last_timestampt":0.0}

def listClocks():
    with __lock:
        return list(__clocks.keys())

def start(name):
    with __lock:
        __clocks[name]['last_timestampt'] = time.perf_counter()

def stop(name):
    with __lock:
        last_ts = __clocks[name]['last_timestampt']
        if last_ts != 0.0:
            passed_time_ms = (time.perf_counter() - last_ts) * 1000
            __clocks[name]['times_ms'] += [passed_time_ms]
            # only keep __CAPACITY of timeslots in the list
            __clocks[name]['times_ms'] = __clocks[name]['times_ms'][:__CAPACITY]
            __clocks[name]['last_timestampt'] = time.perf_counter()

def calculate_avg_OPS(name) -> float:
    with __lock:
        if name not in __clocks.keys() or len(__clocks[name]['times_ms']) == 0:
            return None

        sum_seconds = np.sum(__clocks[name]['times_ms'])/1000
        return len(__clocks[name]['times_ms']) / sum_seconds

def get_std_deviation(name) -> float:
    with __lock:
        if name not in __clocks.keys() or len(__clocks[name]['times_ms']) == 0:
            return None

        return np.std(__clocks[name]['times_ms'])

def get_min_max_values(name):
    with __lock:
        if name not in __clocks.keys() or len(__clocks[name]['times_ms']) == 0:
            return None, None
        
        min = np.min(__clocks[name]['times_ms'])
        max = np.max(__clocks[name]['times_ms'])
        return int(min), int(max)

def reset(name):
    with __lock:
        __clocks[name]['last_timestamp'] = 0.0
        __clocks[name]['times_ms'] = []

def printReport(name) -> str:
    avg_ops = calculate_avg_OPS(name)
    std_div = get_std_deviation(name)
    min, max = get_min_max_values(name)
    
    if avg_ops == None:
        return f'no report for "{name}"'
        
    return (f'=== Clock report for "{name}" ===\r\n'
            f'Average OPS = {avg_ops:.2f}\r\n'
            f'Standard deviation = {std_div:.2f}\r\n'
            f'Minimum processing time = {min:.2f}ms\r\n'
            f'Maximum processing time = {max:.2f}ms\r\n'
             '=== report end ===')
