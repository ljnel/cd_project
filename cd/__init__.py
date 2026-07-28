# Set up library logging - users control output via logging configuration
import logging

logging.getLogger("cd").addHandler(logging.NullHandler())
