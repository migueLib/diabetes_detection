import logging
import sys


def set_logger(level=""):
    """ Function to set up the handle error logging.
    logger (obj) = a logger object
    logLevel (str) = level of information to print out, options are
    {info, debug} [Default: info]
    """

    # Starting a logger
    logger = logging.getLogger()
    error = logging.ERROR

    # Determine log level
    if level == 'debug':
        _level = logging.DEBUG
    else:
        _level = logging.INFO

    # Set the level in logger
    logger.setLevel(_level)

    # Set the log format
    log_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set logger output to STDOUT and STDERR
    log_handler = logging.StreamHandler(stream=sys.stdout)
    err_handler = logging.StreamHandler(stream=sys.stderr)

    # Set logging level for the different output handlers.
    # ERRORs to STDERR, and everything else to STDOUT
    log_handler.setLevel(_level)
    err_handler.setLevel(error)

    # Format the log handlers
    log_handler.setFormatter(log_fmt)
    err_handler.setFormatter(log_fmt)

    # Add handler to the main logger
    logger.addHandler(log_handler)
    logger.addHandler(err_handler)

    return logger
