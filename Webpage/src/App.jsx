import { useState, useEffect } from 'react';
import { Activity, Cpu, HardDrive, Thermometer, Clock, Gamepad2 } from 'lucide-react';

export default function GameStreamMonitor() {
  const [activeStream, setActiveStream] = useState(1);
  const [piMetrics, setPiMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch real-time metrics from Pi via proxy
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        setPiMetrics(data);
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch Pi metrics:', error);
        setLoading(false);
      }
    };

    // Fetch metrics immediately
    fetchMetrics();

    // Set up polling every 1 second
    const interval = setInterval(fetchMetrics, 1000);

    return () => clearInterval(interval);
  }, []);

  // Set mode (1: Baseline, 2: ARM Optimized, 3: ARM Hero)
  const setMode = async (mode) => {
    try {
      if (mode === 1) {
        // Mode 1: Baseline - disable delegate
        await fetch(`/api/control/delegate?enable=false`, { method: 'POST' });
        await fetch(`/api/control/performance_mode?mode=BALANCED`, { method: 'POST' });
      } else if (mode === 2) {
        // Mode 2: ARM Optimized - enable delegate
        await fetch(`/api/control/delegate?enable=true`, { method: 'POST' });
        await fetch(`/api/control/performance_mode?mode=BALANCED`, { method: 'POST' });
      } else if (mode === 3) {
        // Mode 3: ARM Hero - convex hull mode
        await fetch(`/api/control/performance_mode?mode=CONVEX_HULL`, { method: 'POST' });
      }
      // Metrics will update automatically via polling
    } catch (error) {
      console.error('Failed to set mode:', error);
    }
  };

  // Determine current mode based on metrics
  const getCurrentMode = () => {
    if (piMetrics?.model_name?.includes('Classical CV')) return 3;
    if (piMetrics?.delegate_enabled) return 2;
    return 1;
  };

  // Stream configuration
  const currentStream = {
    videoUrl: '/api/stream',
    fps: piMetrics?.fps || 0,
    frameTime: piMetrics?.frame_time_ms || (piMetrics?.fps ? (1000 / piMetrics.fps).toFixed(1) : 0),
    temperature: piMetrics?.cpu_temp_c || 0,
    memoryUsage: piMetrics?.mem_used_mb ? (piMetrics.mem_used_mb / 1024).toFixed(1) : 0,
    inferenceTime: piMetrics?.infer_ms || 0,
    gameScore: piMetrics?.game_score || 0,
    gameMissed: piMetrics?.game_missed || 0,
    gameFruits: piMetrics?.game_fruits || 0,
    gameParticles: piMetrics?.game_particles || 0,
    modelName: piMetrics?.model_name || 'Loading...',
    delegateEnabled: piMetrics?.delegate_enabled || false
  };

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
            ARM Optimization Demo
          </h1>
          <p style={{ color: '#9ca3af' }}>Live comparison: Baseline vs ARM Optimized vs ARM Hero</p>
        </div>

        {/* Mode Selection Buttons */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            onClick={() => setMode(1)}
            disabled={loading}
            style={{
              backgroundColor: getCurrentMode() === 1 ? '#dc2626' : '#374151',
              padding: '0.75rem 1.25rem',
              borderRadius: '0.5rem',
              fontWeight: '600',
              border: getCurrentMode() === 1 ? '2px solid #fca5a5' : '2px solid transparent',
              cursor: loading ? 'not-allowed' : 'pointer',
              color: 'white',
              opacity: loading ? 0.7 : 1,
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>MODE 1</div>
            <div>Baseline</div>
            <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>~9 FPS</div>
          </button>

          <button
            onClick={() => setMode(2)}
            disabled={loading}
            style={{
              backgroundColor: getCurrentMode() === 2 ? '#059669' : '#374151',
              padding: '0.75rem 1.25rem',
              borderRadius: '0.5rem',
              fontWeight: '600',
              border: getCurrentMode() === 2 ? '2px solid #6ee7b7' : '2px solid transparent',
              cursor: loading ? 'not-allowed' : 'pointer',
              color: 'white',
              opacity: loading ? 0.7 : 1,
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>MODE 2</div>
            <div>ARM Optimized</div>
            <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>~16 FPS</div>
          </button>

          <button
            onClick={() => setMode(3)}
            disabled={loading}
            style={{
              backgroundColor: getCurrentMode() === 3 ? '#7c3aed' : '#374151',
              padding: '0.75rem 1.25rem',
              borderRadius: '0.5rem',
              fontWeight: '600',
              border: getCurrentMode() === 3 ? '2px solid #c4b5fd' : '2px solid transparent',
              cursor: loading ? 'not-allowed' : 'pointer',
              color: 'white',
              opacity: loading ? 0.7 : 1,
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>MODE 3</div>
            <div>ARM Hero 🚀</div>
            <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>~176 FPS</div>
          </button>
        </div>
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
                icon={Clock}
                label="Inference Time"
                value={currentStream.inferenceTime}
                unit="ms"
                color="#a855f7"
              />
              <MetricCard
                icon={Thermometer}
                label="Temperature"
                value={currentStream.temperature}
                unit="°C"
                color="#ef4444"
              />
              <MetricCard
                icon={HardDrive}
                label="Memory Usage"
                value={currentStream.memoryUsage}
                unit="GB"
                color="#06b6d4"
              />
              <MetricCard
                icon={Gamepad2}
                label="Game Score"
                value={currentStream.gameScore}
                unit="pts"
                color="#22c55e"
              />
              <MetricCard
                icon={Activity}
                label="Fruits Active"
                value={currentStream.gameFruits}
                unit=""
                color="#f97316"
              />
              <MetricCard
                icon={Activity}
                label="Missed"
                value={currentStream.gameMissed}
                unit=""
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
              <div>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '600' }}>
                  Fruit Ninja AR - Mode {getCurrentMode()}
                </h2>
                <p style={{ color: '#9ca3af', fontSize: '0.875rem' }}>
                  {currentStream.modelName}
                </p>
              </div>
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
              overflow: 'hidden',
              border: '1px solid #374151',
              position: 'relative'
            }}>
              {loading ? (
                <div style={{
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#9ca3af'
                }}>
                  Loading stream...
                </div>
              ) : (
                <>
                  <img
                    src={currentStream.videoUrl}
                    alt="Fruit Ninja AR Stream"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                      backgroundColor: '#000'
                    }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'flex';
                    }}
                  />
                  {/* Performance overlay - rendered on frontend for sharp text */}
                  <div style={{
                    position: 'absolute',
                    bottom: '10px',
                    left: '10px',
                    color: 'white',
                    fontSize: '14px',
                    fontFamily: 'monospace',
                    backgroundColor: 'rgba(0, 0, 0, 0.6)',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    pointerEvents: 'none',
                    whiteSpace: 'nowrap'
                  }}>
                    FPS: {currentStream.fps} | Frame: {currentStream.frameTime}ms | Fruits: {currentStream.gameFruits} | Particles: {currentStream.gameParticles}
                  </div>
                </>
              )}
              <div style={{
                width: '100%',
                height: '100%',
                display: 'none',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ef4444',
                flexDirection: 'column',
                gap: '0.5rem'
              }}>
                <p>Stream Offline</p>
                <p style={{ fontSize: '0.875rem', color: '#9ca3af' }}>
                  Check Pi connection
                </p>
              </div>
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