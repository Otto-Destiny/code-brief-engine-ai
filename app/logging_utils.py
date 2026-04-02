from __future__ import annotations

import json
import logging
from typing import Any


def configure_logging() -> None:
    if getattr(configure_logging, "_configured", False):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    configure_logging._configured = True


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, sort_keys=True))
