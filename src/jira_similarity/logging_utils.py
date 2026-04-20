from __future__ import annotations

import logging
import os


def configure_logging(level_name: str | None = None) -> None:
    effective_level_name = (level_name or os.getenv("JIRA_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, effective_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
