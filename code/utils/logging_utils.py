"""AIND log formatters, filters, and handler setup for use with logging.yml dictConfig."""

import datetime
import json
import logging
import logging.config
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.yml")


def setup_logging(log_stream: str, **fields) -> None:
    """Load logging.yml and configure logging for the application.

    Also adds a CloudWatch handler so logs are sent to the default log group.
    Requires valid AWS credentials at runtime.

    Parameters
    ----------
    log_stream : str
        CloudWatch log stream name, e.g. the process or capsule name.
    **fields
        Arbitrary context fields to inject into every log record, e.g.
        ``acquisition_name="123456_2026-03-23_10-00-00"``,
        ``process_name="my-capsule"``.
    """
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        config["filters"]["aind_fields"].update(fields)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.DEBUG)
        logging.warning("logging.yml not found — falling back to basicConfig")

    add_cloudwatch_handler(log_stream=log_stream)


def add_cloudwatch_handler(log_stream: str) -> None:
    """Programmatically add a CloudWatch handler to the root logger."""
    try:
        import watchtower
    except ImportError:
        logging.warning("watchtower not installed — skipping CloudWatch handler")
        return

    handler = watchtower.CloudWatchLogHandler(
        log_stream_name=log_stream,
        log_group="aind/internal-logs",
    )

    root_logger = logging.getLogger()
    formatter = AindJsonFormatter()
    for existing_handler in root_logger.handlers:
        if isinstance(existing_handler.formatter, AindJsonFormatter):
            formatter = existing_handler.formatter
        for f in existing_handler.filters:
            if isinstance(f, AindContextFilter):
                handler.addFilter(f)
                break

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class AindJsonFormatter(logging.Formatter):
    """Formats log records as JSON with AIND standard fields."""

    _DEFAULT_FIELDS = ["timestamp", "level", "message", "acquisition_name"]

    def __init__(self, fields=None):
        super().__init__()
        self.fields = fields or self._DEFAULT_FIELDS

    def format(self, record):
        log_entry = {}
        for field in self.fields:
            if field == "timestamp":
                log_entry["timestamp"] = (
                    datetime.datetime.fromtimestamp(
                        record.created, tz=datetime.timezone.utc
                    )
                    .astimezone()
                    .isoformat(timespec="milliseconds")
                )
            elif field == "level":
                log_entry["level"] = record.levelname
            elif field == "message":
                log_entry["message"] = record.getMessage()
            else:
                value = getattr(record, field, None)
                if value is not None and value != "undefined":
                    log_entry[field] = value
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class AindContextFilter(logging.Filter):
    """Injects arbitrary context fields into every log record."""

    def __init__(self, **fields):
        super().__init__()
        self.fields = fields

    def filter(self, record):
        for key, value in self.fields.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class LoggingStream:
    """File-like stream that routes write() calls through a logger.

    Use as a stdout/stderr replacement to capture print() output from
    third-party libraries into structured JSON logs.

    Usage::

        old_stdout = sys.stdout
        sys.stdout = LoggingStream(logger, logging.INFO)
        try:
            third_party_function()
        finally:
            sys.stdout = old_stdout
    """

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, msg):
        if msg.strip():
            self.logger.log(self.level, msg.strip())

    def flush(self):
        pass
