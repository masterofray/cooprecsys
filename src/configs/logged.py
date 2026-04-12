#!/usr/bin/env python3

'''
Module Name : Logging General usage
Description : Handles the logging area and configs.
author      : Aryanto
compiler    : python 3.10
date        : 20260324
Contact     : aryanto.dandan@gmail.com
'''


import os
import sys
import logging
from datetime import datetime
from configparser import ConfigParser

def setup_logging():
    # 1. Resolve Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = datetime.now().strftime('%Y%m%d')
    log_dir = os.path.join(base_dir, 'IO', 'LOG', date_str)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Load Config
    config = ConfigParser()
    config.read(os.path.join(base_dir, 'Conf.ini'))
    
    log_name = config.get('DEFAULT', 'LOG_FILE', fallback='app.log')
    log_path = os.path.join(log_dir, log_name)

    # 3. Configure Logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)