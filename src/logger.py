import os
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',level='INFO')

class CustomLogger:
    def __init__(self):
        self.logger = None

    def log_struct(self, message, severity="DEFAULT"):
        logging.info(f"[{severity}] {message}")
