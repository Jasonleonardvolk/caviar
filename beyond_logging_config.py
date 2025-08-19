#!/usr/bin/env python3
"""
beyond_logging_config.py - Configure structured logging for Beyond Metacognition
Integrates with journald for systemd and formats for Loki ingestion
"""

import logging
import json
from datetime import datetime
from pathlib import Path
import sys

class JournaldStructuredFormatter(logging.Formatter):
    """
    Format logs for journald with structured fields
    These can be queried in Loki with LogQL
    """
    
    def format(self, record):
        # Base message with Beyond tags
        message = super().format(record)
        
        # Add structured fields for journald
        # These become labels in Loki
        extras = {
            'COMPONENT': 'beyond_metacognition',
            'SUBSYSTEM': self._get_subsystem(record.name),
            'SEVERITY': record.levelname,
            'PRIORITY': self._level_to_priority(record.levelno),
        }
        
        # Add Beyond-specific fields if present
        if hasattr(record, 'spectral_state'):
            extras['SPECTRAL_STATE'] = record.spectral_state
            
        if hasattr(record, 'lambda_max'):
            extras['LAMBDA_MAX'] = str(record.lambda_max)
            
        if hasattr(record, 'creative_mode'):
            extras['CREATIVE_MODE'] = record.creative_mode
            
        if hasattr(record, 'novelty_score'):
            extras['NOVELTY_SCORE'] = str(record.novelty_score)
        
        # Format for journald (key=value pairs)
        structured = ' '.join(f'{k}={v}' for k, v in extras.items())
        
        # Prepend tags for easy grepping
        return f"[BEYOND] [SPECTRAL] {message} {structured}"
    
    def _get_subsystem(self, logger_name):
        """Map logger name to subsystem"""
        subsystem_map = {
            'origin_sentry': 'ORIGIN',
            'braid_buffers': 'BRAID',
            'observer_synthesis': 'OBSERVER',
            'creative_feedback': 'CREATIVE',
            'braid_aggregator': 'AGGREGATOR'
        }
        
        for key, value in subsystem_map.items():
            if key in logger_name:
                return value
        return 'CORE'
    
    def _level_to_priority(self, level):
        """Convert Python log level to syslog priority"""
        # Syslog priorities for journald
        priority_map = {
            logging.CRITICAL: 2,  # LOG_CRIT
            logging.ERROR: 3,     # LOG_ERR
            logging.WARNING: 4,   # LOG_WARNING
            logging.INFO: 6,      # LOG_INFO
            logging.DEBUG: 7      # LOG_DEBUG
        }
        return priority_map.get(level, 6)

class BeyondLoggingConfig:
    """Configure logging for Beyond Metacognition components"""
    
    @staticmethod
    def setup_logging(log_level='INFO', enable_file_logging=True):
        """
        Setup structured logging for all Beyond components
        """
        # Create formatters
        journald_formatter = JournaldStructuredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler (for systemd/journald)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(journald_formatter)
        console_handler.setLevel(getattr(logging, log_level))
        
        # Configure all Beyond loggers
        beyond_loggers = [
            'alan_backend.origin_sentry',
            'alan_backend.braid_aggregator',
            'python.core.braid_buffers',
            'python.core.observer_synthesis',
            'python.core.creative_feedback',
            'python.core.chaos_control_layer',
            'eigensentry_guard'
        ]
        
        for logger_name in beyond_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(console_handler)
            logger.propagate = False
        
        # Optional file logging for debugging
        if enable_file_logging:
            log_dir = Path("/opt/tori/logs/beyond")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"beyond_{datetime.now().strftime('%Y%m%d')}.log",
                encoding='utf-8'
            )
            file_handler.setFormatter(journald_formatter)
            file_handler.setLevel(logging.DEBUG)
            
            for logger_name in beyond_loggers:
                logger = logging.getLogger(logger_name)
                logger.addHandler(file_handler)
    
    @staticmethod
    def create_loki_config():
        """
        Create Promtail config for shipping Beyond logs to Loki
        """
        config = """
# promtail-beyond.yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions-beyond.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  # Scrape journald for Beyond logs
  - job_name: beyond_metacognition
    journal:
      matches: "_SYSTEMD_UNIT=tori-api.service,tori-braid-aggregator.service,tori-beyond-monitor.service"
      labels:
        job: beyond_metacognition
        host: ${HOSTNAME}
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
      - source_labels: ['__journal_component']
        target_label: 'component'
      - source_labels: ['__journal_subsystem']
        target_label: 'subsystem'
      - source_labels: ['__journal_severity']
        target_label: 'severity'
    pipeline_stages:
      # Extract Beyond-specific fields
      - regex:
          expression: '\\[BEYOND\\] \\[SPECTRAL\\] (?P<message>.*?) LAMBDA_MAX=(?P<lambda_max>[0-9.]+)'
      - labels:
          lambda_max:
      - regex:
          expression: 'CREATIVE_MODE=(?P<creative_mode>\\w+)'
      - labels:
          creative_mode:
      - regex:
          expression: 'NOVELTY_SCORE=(?P<novelty>[0-9.]+)'
      - labels:
          novelty_score:
          
  # Also scrape file logs if enabled
  - job_name: beyond_files
    static_configs:
      - targets:
          - localhost
        labels:
          job: beyond_metacognition
          __path__: /opt/tori/logs/beyond/*.log
    pipeline_stages:
      - multiline:
          firstline: '^\\d{4}-\\d{2}-\\d{2}'
      - regex:
          expression: '\\[(?P<tag1>\\w+)\\] \\[(?P<tag2>\\w+)\\]'
      - labels:
          tag1:
          tag2:
"""
        return config
    
    @staticmethod
    def create_loki_alerts():
        """
        Create Loki alerting rules for Beyond Metacognition
        """
        rules = """
# beyond_alerts.yaml
groups:
  - name: beyond_metacognition
    rules:
      - alert: HighLambdaMax
        expr: |
          avg_over_time({job="beyond_metacognition"} |= "LAMBDA_MAX" | regexp "LAMBDA_MAX=(?P<value>[0-9.]+)" | unwrap value [5m]) > 0.08
        for: 3m
        labels:
          severity: critical
          component: beyond_metacognition
        annotations:
          summary: "High λ_max sustained for 3 minutes"
          description: "Lambda max {{ $value }} exceeds threshold, auto-rollback may trigger"
          
      - alert: DimensionalExpansionSpike
        expr: |
          rate({job="beyond_metacognition"} |= "Dimensional expansion detected" [5m]) > 0.1
        for: 1m
        labels:
          severity: warning
          component: origin_sentry
        annotations:
          summary: "High rate of dimensional expansions"
          description: "{{ $value }} expansions per second detected"
          
      - alert: CreativeEmergencyMode
        expr: |
          {job="beyond_metacognition"} |= "CREATIVE_MODE=emergency"
        for: 0m
        labels:
          severity: warning
          component: creative_feedback
        annotations:
          summary: "Creative feedback in emergency mode"
          description: "System entered emergency damping mode"
          
      - alert: ReflexiveOscillation
        expr: |
          {job="beyond_metacognition"} |= "Reflexive oscillation detected"
        for: 0m
        labels:
          severity: warning
          component: observer_synthesis
        annotations:
          summary: "Reflexive oscillation detected"
          description: "Self-measurement entering oscillatory pattern"
"""
        return rules

def generate_rsyslog_config():
    """
    Generate rsyslog config for Beyond logs
    """
    config = """
# /etc/rsyslog.d/49-beyond.conf
# Forward Beyond Metacognition logs

# Template for structured logging
template(name="BeyondFormat" type="string"
  string="%timegenerated% %hostname% %syslogtag% [BEYOND] %msg%\\n"
)

# Filter Beyond logs
if $programname == 'tori-api' and $msg contains '[BEYOND]' then {
    action(type="omfile" file="/var/log/tori/beyond.log" template="BeyondFormat")
    
    # Also forward to remote syslog if configured
    # action(type="omfwd" target="syslog.example.com" port="514" protocol="tcp")
}

if $programname == 'tori-braid' then {
    action(type="omfile" file="/var/log/tori/beyond-braid.log" template="BeyondFormat")
}

if $programname == 'tori-monitor' then {
    action(type="omfile" file="/var/log/tori/beyond-monitor.log" template="BeyondFormat")
}

# Stop processing these messages
& stop
"""
    return config

# Usage in TORI components
def setup_component_logger(component_name, extra_fields=None):
    """
    Setup logger for a specific Beyond component
    
    Usage:
        logger = setup_component_logger('origin_sentry')
        logger.info("Dimensional expansion detected", 
                    extra={'lambda_max': 0.05, 'spectral_state': 'global'})
    """
    logger = logging.getLogger(f'alan_backend.{component_name}')
    
    # Add component-specific handlers if needed
    if extra_fields:
        # Create adapter to add default fields
        class BeyondLogAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add default extra fields
                extra = kwargs.get('extra', {})
                extra.update(extra_fields)
                kwargs['extra'] = extra
                return msg, kwargs
        
        return BeyondLogAdapter(logger, {})
    
    return logger

if __name__ == "__main__":
    # Setup logging when imported
    BeyondLoggingConfig.setup_logging()
    
    # Generate config files
    loki_config = BeyondLoggingConfig.create_loki_config()
    with open("promtail-beyond.yaml", "w") as f:
        f.write(loki_config)
    print("✅ Created promtail-beyond.yaml")
    
    alerts = BeyondLoggingConfig.create_loki_alerts()
    with open("beyond_alerts.yaml", "w") as f:
        f.write(alerts)
    print("✅ Created beyond_alerts.yaml")
    
    rsyslog = generate_rsyslog_config()
    with open("49-beyond.conf", "w") as f:
        f.write(rsyslog)
    print("✅ Created 49-beyond.conf")
