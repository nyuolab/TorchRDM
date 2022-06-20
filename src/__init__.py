import logging
import logging.config

# https://stackoverflow.com/questions/52175730/python-question-on-logging-in-init-py

# Create the Logger
loggers = logging.getLogger(__name__)
loggers.setLevel(logging.DEBUG)

# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Create the Handler for logging data to a file
logger_handler = logging.StreamHandler()
logger_handler.setLevel(logging.DEBUG)

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
loggers.addHandler(logger_handler)
