"""
MagicQuant Structured Logging - structlog configuration.

Provides a consistent logging interface with structured fields:
  event, stage, tensor_name, progress, etc.
"""

from typing import Optional

import structlog


def configure_logging(*, verbose: bool = True, json_output: bool = False) -> None:
    """Configure structlog for MagicQuant.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.
        json_output: If True, render logs as JSON; otherwise human-readable.
    """
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally named.

    Args:
        name: Logger name, typically the module ``__name__``.

    Returns:
        A structlog BoundLogger instance.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger
