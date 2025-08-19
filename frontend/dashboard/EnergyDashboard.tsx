import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface EnergyMetrics {
  timestamp: number;
  totalEnergy: number;
  kineticEnergy: number;
  potentialEnergy: number;
  batteryCharge: number;
  conservationError: number;
}

export const EnergyDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<EnergyMetrics[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket metrics channel
    const websocket = new WebSocket('ws://localhost:8766/metrics');
    
    websocket.onopen = () => {
      setIsConnected(true);
      console.log('Connected to energy metrics stream');
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'energy_update') {
        setMetrics(prev => {
          const updated = [...prev, data.metrics];
          // Keep only last 500 data points
          return updated.slice(-500);
        });
      }
    };

    websocket.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from metrics stream');
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setWs(websocket);

    return () => {
      websocket.close();
    };
  }, []);

  const chartData = {
    labels: metrics.map(m => new Date(m.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Total Energy',
        data: metrics.map(m => m.totalEnergy),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        tension: 0.1
      },
      {
        label: 'Kinetic Energy',
        data: metrics.map(m => m.kineticEnergy),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        tension: 0.1
      },
      {
        label: 'Potential Energy',
        data: metrics.map(m => m.potentialEnergy),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.1
      },
      {
        label: 'Battery Charge',
        data: metrics.map(m => m.batteryCharge),
        borderColor: 'rgb(255, 206, 86)',
        backgroundColor: 'rgba(255, 206, 86, 0.5)',
        tension: 0.1,
        yAxisID: 'y1'
      }
    ]
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Soliton Memory Energy Dashboard'
      },
      tooltip: {
        callbacks: {
          afterLabel: (context) => {
            const index = context.dataIndex;
            const error = metrics[index]?.conservationError || 0;
            return `Conservation Error: ${error.toExponential(2)}`;
          }
        }
      }
    },
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Energy (J)'
        }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: {
          display: true,
          text: 'Battery Charge (%)'
        },
        grid: {
          drawOnChartArea: false,
        },
      },
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time'
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45,
          maxTicksLimit: 20
        }
      }
    }
  };

  const latestMetric = metrics[metrics.length - 1];
  const conservationStatus = latestMetric?.conservationError < 1e-5 ? 'GOOD' : 'WARNING';

  return (
    <div className="energy-dashboard">
      <div className="dashboard-header">
        <h2>Energy Conservation Monitor</h2>
        <div className="connection-status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      <div className="metrics-summary">
        <div className="metric-card">
          <h3>Total Energy</h3>
          <p>{latestMetric?.totalEnergy.toFixed(6) || '---'} J</p>
        </div>
        <div className="metric-card">
          <h3>Conservation Error</h3>
          <p className={`conservation-${conservationStatus.toLowerCase()}`}>
            {latestMetric?.conservationError.toExponential(2) || '---'}
          </p>
          <span className="status-label">{conservationStatus}</span>
        </div>
        <div className="metric-card">
          <h3>Battery Level</h3>
          <p>{latestMetric?.batteryCharge.toFixed(1) || '---'}%</p>
        </div>
      </div>

      <div className="chart-container">
        {metrics.length > 0 && <Line options={options} data={chartData} />}
      </div>

      <style jsx>{`
        .energy-dashboard {
          padding: 20px;
          background: #f5f5f5;
          border-radius: 8px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .status-indicator {
          width: 12px;
          height: 12px;
          border-radius: 50%;
          display: inline-block;
        }

        .status-indicator.connected {
          background-color: #4caf50;
        }

        .status-indicator.disconnected {
          background-color: #f44336;
        }

        .metrics-summary {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
          margin-bottom: 24px;
        }

        .metric-card {
          background: white;
          padding: 16px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metric-card h3 {
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #666;
        }

        .metric-card p {
          margin: 0;
          font-size: 24px;
          font-weight: bold;
        }

        .conservation-good {
          color: #4caf50;
        }

        .conservation-warning {
          color: #ff9800;
        }

        .status-label {
          font-size: 12px;
          color: #666;
        }

        .chart-container {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
      `}</style>
    </div>
  );
};

export default EnergyDashboard;
