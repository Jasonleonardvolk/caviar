# MCP Metacognitive Web Interface

A real-time dashboard for monitoring and controlling TORI's metacognitive systems.

## Features

- 🎯 **Real-time Metrics Dashboard** - Monitor response times, error rates, and system health
- 🧠 **Kaizen Insights Viewer** - Browse and analyze continuous improvement insights  
- 📊 **Prometheus Metrics Export** - Full metrics endpoint for external monitoring
- 🔄 **WebSocket Updates** - Live updates without page refresh
- 🎮 **Control Panel** - Start/stop Kaizen, trigger analyses
- 📚 **API Documentation** - Interactive API docs via FastAPI

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web interface:
```bash
python run_web.py --port 8088
```

3. Open your browser to http://localhost:8088

## Available Endpoints

- `/` - Main dashboard interface
- `/api/status` - System status JSON
- `/api/insights` - Recent Kaizen insights
- `/api/kaizen/start` - Start continuous improvement
- `/api/kaizen/stop` - Stop continuous improvement
- `/api/kaizen/analyze` - Trigger immediate analysis
- `/kaizen/metrics` - Prometheus metrics endpoint
- `/docs` - Interactive API documentation
- `/ws` - WebSocket for real-time updates

## Configuration

The web interface respects all Kaizen environment variables:

```bash
# Enable Prometheus metrics
export KAIZEN_ENABLE_PROMETHEUS=true

# Set analysis interval (seconds)
export KAIZEN_ANALYSIS_INTERVAL=3600

# Enable auto-start
export KAIZEN_ENABLE_AUTO_START=true
```

## Dashboard Components

### Status Panel
- Kaizen running status
- Control buttons (Start/Stop/Analyze)

### Metrics Cards
- Average Response Time
- Error Rate
- Total Queries Processed
- Active Insights Count
- Knowledge Base Size

### Insights Feed
- Recent insights with confidence scores
- Insight types and timestamps
- Recommendations for improvement

### System Log
- Real-time activity log
- WebSocket connection status
- Analysis results

## Development

### Running in Development Mode
```bash
uvicorn run_web:app --reload --port 8088
```

### Running Tests
```bash
pytest test_web.py -v
```

## Integration with Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'mcp_kaizen'
    static_configs:
      - targets: ['localhost:8088']
    metrics_path: '/kaizen/metrics'
    scrape_interval: 30s
```

## Troubleshooting

### Kaizen Not Available
If you see "Kaizen not available" errors:
1. Ensure the parent MCP modules are in your Python path
2. Check that all dependencies are installed
3. Verify the agents directory structure

### WebSocket Connection Issues
If real-time updates aren't working:
1. Check browser console for errors
2. Ensure no firewall/proxy is blocking WebSocket connections
3. Try a different browser

### Port Already in Use
If port 8088 is taken:
```bash
python run_web.py --port 8089
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   Web Browser   │────▶│  FastAPI Server  │
└─────────────────┘     └──────────────────┘
         │                       │
         │                       ▼
         │              ┌──────────────────┐
         │              │ Kaizen Engine    │
         │              └──────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│   WebSocket     │     │ Prometheus       │
│   Updates       │     │ Metrics          │
└─────────────────┘     └──────────────────┘
```

## Future Enhancements

- [ ] Historical metrics graphs
- [ ] Insight application tracking
- [ ] Multi-agent status monitoring
- [ ] Export insights to CSV/JSON
- [ ] Dark mode theme
- [ ] Mobile-responsive design

## License

Part of the TORI MCP Metacognitive System
