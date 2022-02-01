import pyaudio
import json

p = pyaudio.PyAudio()
device_list = []
for d in range(p.get_device_count()):
    device_list.append(p.get_device_info_by_index(d))

f = open("device_list.json", "w")
f.write(json.dumps(device_list, indent=4))
f.close()