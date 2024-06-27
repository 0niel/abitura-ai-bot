import logging


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Function to setup a logger with the given name and log file."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Setup the logger for the chatbot
logger = setup_logger("chatbot_logger", "chatbot.log")
