import re, subprocess
from threading import Thread, Lock

class PiMetrics(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.daemon = True
        self.__cpu_temp = 0.0
        self.__cpu_temp_lock = Lock()
        self.__cpu_clock = 0.0
        self.__cpu_clock_lock = Lock()
        self.__mem_allocation = 0.0
        self.__mem_allocation_lock = Lock()

    def __update_cpu_temp(self):
        temp = 0.0
        err, msg = subprocess.getstatusoutput('vcgencmd measure_temp')
        if not err:
            m = re.search(r'-?\d\.?\d*', msg)   # a solution with a  regex
            try:
                temp = float(m.group())
            except ValueError: # catch only error needed
                pass
        with self.__cpu_temp_lock:
            self.__cpu_temp = temp

    def __update_cpu_clock(self):
        #TODO: impelement
        with self.__cpu_clock_lock:
            self.__cpu_clock = 40.0

    def __update_mem_allocation(self):
        #TODO: impelement
        with self.__mem_allocation_lock:
            self.__mem_allocation = 50.0

    def run(self):
        while self.isAlive:
            self.__update_cpu_clock()
            self.__update_cpu_temp()
            self.__update_mem_allocation()

    def get_cpu_temp(self):
        with self.__cpu_temp_lock:
            return self.__cpu_temp

    def get_cpu_clock(self):
        with self.__cpu_clock_lock:
            return self.__cpu_clock

    def get_mem_allocation(self):
        with self.__mem_allocation_lock:
            return self.__mem_allocation


