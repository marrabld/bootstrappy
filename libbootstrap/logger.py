__author__ = 'marrabld'

import logging.config
import os

log_conf_file = os.path.join(os.path.dirname(__file__), 'logging.conf')
logging.config.fileConfig(log_conf_file)

# create logger
logger = logging.getLogger('libbootstrappy')

# log_file = 'libbootstrappy.log'
# basicConfig(filename=log_file, format='%(asctime)s :: %(levelname)s :: %(message)s', level=DEBUG,
#                   datefmt='%m/%d/%Y %I:%M:%S %p')


def clear_log():
    """
    This method will clear the log file by reopening the file for writing.
    """
    with open('libbootstrappy.log', 'w'):
        pass


def clear_err():
    """
    This method will clear the log file by reopening the file for writing.
    """

    with open('libbootstrappy.err', 'w'):
        pass