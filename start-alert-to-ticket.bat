@echo off
:: Start the Alertmanager to Kaizen ticket bridge
echo Starting Alertmanager to Kaizen ticket bridge...
echo This service listens for alerts from Prometheus Alertmanager
echo and automatically creates Kaizen tickets for issues.

:: Change to the registry directory
cd registry\kaizen

:: Start the service
python alert_to_ticket.py --port 9099

echo.
echo Server stopped.
