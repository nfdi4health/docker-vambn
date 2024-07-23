import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(level: int, log_file: Optional[Path] = None) -> None:
    """
    Setup logging to stdout or a specified file.

    Args:
        level (int): The logging level.
        log_file (Optional[Path], optional): The file where logs should be saved. Defaults to None.

    """
    if log_file is None:
        logging.basicConfig(
            stream=sys.stdout,
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        logger.info("Logging to stdout")
    else:
        log_file.parent.mkdir(exist_ok=True, parents=True)
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=10_000_000,
            backupCount=10,
        )
        # Create a format with the file name in the log
        formatter = logging.Formatter(
            "[%(asctime)s] {%(module)s:%(lineno)d} %(levelname)s - %(message)s",
            "%m-%d %H:%M:%S",
        )

        file_handler.setFormatter(formatter)
        logging.basicConfig(
            level=level,
            handlers=[file_handler],
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        try:
            pl_logger = logging.getLogger("lightning.pytorch")
            pl_logger.handlers.clear()
            pl_logger.setLevel(level=level)
            pl_logger.addHandler(file_handler)
            pl_logger.info("Logging Lightning to file")
        except Exception as e:
            logger.error(f"Could not setup Lightning logging: {e}")

        logger.info(f"Logging to {log_file}")
