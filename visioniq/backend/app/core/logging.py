import sys
import logging
import json
import time
from typing import Any, Dict
from backend.app.core.config import settings

class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON for observability and trace auditing."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if hasattr(record, "trace_id"):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, "agent"):
            log_entry["agent"] = record.agent
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def setup_logging():
    """Initializes standard console and structured log handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.LOG_LEVEL.upper())

    # Clear existing handlers
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    if settings.DEBUG:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s [%(name)s:%(lineno)d] %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        formatter = StructuredJsonFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger("visioniq")
