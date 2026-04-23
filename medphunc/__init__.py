import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

def enable_logging(level=logging.INFO):
    """Quick setup for interactive/notebook use."""
    logging.basicConfig(level=level, format='%(name)s - %(levelname)s - %(message)s')
#enable_logging()