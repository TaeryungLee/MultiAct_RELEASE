import logging
import os
from utils.dir import mkdir
from colorlog import ColoredFormatter

class colorLogger():
    def __init__(self, path, filename="train.log"):
        self._logger = logging.getLogger(filename)
        mkdir(path)

        self._logger.setLevel(logging.DEBUG)
        consoleformatter = ColoredFormatter(
            fmt="%(log_color)s[%(asctime)s] %(name)s   %(message)s", 
            datefmt="%m-%d %H:%M:%S",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'white,bold',
                'WARNING':  'yellow',
                'ERROR':    'red,bold',
                'CRITICAL': 'red,bg_white'
            })
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(name)s   %(message)s", 
            datefmt="%m-%d %H:%M:%S"
            )

        console = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(path, filename))
        
        # Console: INFO or higher
        # File: DEBUG or higher
        console.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)

        console.setFormatter(consoleformatter)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(console)
        self._logger.addHandler(file_handler)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(str(msg))

    def error(self, msg):
        self._logger.error(str(msg))

    def critical(self, msg):
        self._logger.critical(str(msg))
