import logging
import json
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'message': record.getMessage(),
        }
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        return json.dumps(log_record, ensure_ascii=False)

def setup_json_logger(name='prajna'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = setup_json_logger()
