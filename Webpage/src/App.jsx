import { useState } from 'react';
import { Activity, Cpu, HardDrive, Thermometer, Clock, Gamepad2 } from 'lucide-react';

export default function GameStreamMonitor() {
  const [activeStream, setActiveStream] = useState(1);

  // Placeholder data for Stream 1
  const stream1Data = {
    videoUrl: 'https://www.youtube.com/watch?v=uZkaJ3e9nfY',
    fps: 144,
    frameTime: 6.9,
    gpuUtilization: 78,
    cpuUtilization: 45,
    temperature: 72,
    memoryUsage: 8.4,
    diskLoad: 12,
    inputLatency: 3.2
  };

  // Placeholder data for Stream 2
  const stream2Data = {
    videoUrl: 'https://via.placeholder.com/1280x720/2e1a1a/eee?text=Stream+2',
    fps: 120,
    frameTime: 8.3,
    gpuUtilization: 82,
    cpuUtilization: 52,
    temperature: 75,
    memoryUsage: 9.1,
    diskLoad: 15,
    inputLatency: 4.1
  };

  const currentStream = activeStream === 1 ? stream1Data : stream2Data;

  const MetricCard = ({ icon: Icon, label, value, unit, color }) => (
    <div style={{
      backgroundColor: '#1f2937',
      borderRadius: '0.5rem',
      padding: '0.75rem',
      border: '1px solid #374151'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <Icon style={{ width: '1rem', height: '1rem', color: color }} />
        <span style={{ color: '#9ca3af', fontSize: '0.875rem' }}>{label}</span>
      </div>
      <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'white' }}>
        {value}
        <span style={{ fontSize: '0.875rem', color: '#9ca3af', marginLeft: '0.25rem' }}>{unit}</span>
      </div>
    </div>
  );

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#111827', 
      color: 'white', 
      padding: '1.5rem' 
    }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '1.5rem', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between' 
      }}>
        <div>
          <h1 style={{ fontSize: '2.25rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
            Game Stream Monitor
          </h1>
          <p style={{ color: '#9ca3af' }}>Real-time performance metrics and video streams</p>
        </div>
        <button
          onClick={() => setActiveStream(activeStream === 1 ? 2 : 1)}
          style={{
            backgroundColor: '#2563eb',
            padding: '0.75rem 1.5rem',
            borderRadius: '0.5rem',
            fontWeight: '600',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: 'white'
          }}
          onMouseOver={(e) => e.target.style.backgroundColor = '#1d4ed8'}
          onMouseOut={(e) => e.target.style.backgroundColor = '#2563eb'}
        >
          <Gamepad2 style={{ width: '1.25rem', height: '1.25rem' }} />
          {activeStream === 1 ? "Baseline" : "ARM Optimized"}
        </button>
      </div>

      {/* Main Content */}
      <div style={{ display: 'flex', gap: '1.5rem' }}>
        {/* Metrics Section - Left Side */}
        <div style={{ width: '20%', minWidth: '250px' }}>
          <div style={{
            backgroundColor: '#1f2937',
            borderRadius: '0.5rem',
            padding: '1rem',
            border: '1px solid #374151'
          }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '1rem' }}>
              Performance Metrics
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <MetricCard 
                icon={Activity} 
                label="FPS" 
                value={currentStream.fps} 
                unit="fps"
                color="#22c55e"
              />
              <MetricCard 
                icon={Clock} 
                label="Frame Time" 
                value={currentStream.frameTime} 
                unit="ms"
                color="#3b82f6"
              />
              <MetricCard 
                icon={Activity} 
                label="GPU Usage" 
                value={currentStream.gpuUtilization} 
                unit="%"
                color="#a855f7"
              />
              <MetricCard 
                icon={Cpu} 
                label="CPU Usage" 
                value={currentStream.cpuUtilization} 
                unit="%"
                color="#eab308"
              />
              <MetricCard 
                icon={Thermometer} 
                label="Temperature" 
                value={currentStream.temperature} 
                unit="°C"
                color="#ef4444"
              />
              <MetricCard 
                icon={Activity} 
                label="Memory Usage" 
                value={currentStream.memoryUsage} 
                unit="GB"
                color="#06b6d4"
              />
              <MetricCard 
                icon={HardDrive} 
                label="Disk Load" 
                value={currentStream.diskLoad} 
                unit="%"
                color="#f97316"
              />
              <MetricCard 
                icon={Gamepad2} 
                label="Input Latency" 
                value={currentStream.inputLatency} 
                unit="ms"
                color="#ec4899"
              />
            </div>
          </div>
        </div>

        {/* Video Stream Section - Right Side */}
        <div style={{ flex: 1 }}>
          <div style={{
            backgroundColor: '#1f2937',
            borderRadius: '0.5rem',
            padding: '1rem',
            border: '1px solid #374151'
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'space-between', 
              marginBottom: '1rem' 
            }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '600' }}>{activeStream === 1 ? "Baseline" : "ARM Optimized"}</h2>
              <span style={{
                padding: '0.25rem 0.75rem',
                backgroundColor: '#dc2626',
                borderRadius: '9999px',
                fontSize: '0.875rem',
                fontWeight: '500',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <span style={{
                  width: '0.5rem',
                  height: '0.5rem',
                  backgroundColor: 'white',
                  borderRadius: '9999px',
                  animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
                }}></span>
                LIVE
              </span>
            </div>
            <div style={{
              aspectRatio: '16/9',
              backgroundColor: '#111827',
              borderRadius: '0.5rem',
              overflow: 'hidden'
            }}>
              <img 
                src={currentStream.videoUrl} 
                alt={`Stream ${activeStream}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
}