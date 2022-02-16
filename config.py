import yaml
import os
import sys
from rich import print
from logger import log

config_path = os.getenv('CONFIG_PATH')
config_path = config_path if config_path is not None else "config.yaml"

try:
    with open(config_path, "r") as config_yaml:
            configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
            for c in configs:
                if c['use']:
                    CONFIG = c
                    break
            log.info(f"Config read from '{config_path}'")
            print(CONFIG)
            AUDIO = CONFIG['audio']
            MFCC = CONFIG['mfcc']
            MODEL = CONFIG['model']
            config_yaml.close()
except:
    log.exception("Export path to config file to $CONFIG_PATH environment variable")
    sys.exit()
    