import re, subprocess
 
def get_CPU_temp() -> float:
    temp = 0.0
    err, msg = subprocess.getstatusoutput('vcgencmd measure_temp')
    if not err:
        m = re.search(r'-?\d\.?\d*', msg)   # a solution with a  regex
        try:
            temp = float(m.group())
        except ValueError: # catch only error needed
            pass
    return temp

def get_CPU_clock() -> float:
    #TODO: impelement
    return 0.0

def get_mem_alocation() -> float:
    #TODO: impelement
    return 0.0
