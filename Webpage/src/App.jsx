import { useState, useEffect } from 'react';
import { Activity, Cpu, HardDrive, Thermometer, Clock, Gamepad2 } from 'lucide-react';

export default function GameStreamMonitor() {
  const [activeStream, setActiveStream] = useState(1);
  const [piMetrics, setPiMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  // Combined Palette
  const colors = {
    // Dark Palette for Backgrounds (Blue/Navy)
    mainBackground: '#000D1B', // Darkest Navy/Black (Used as main UI background)
    sectionBackground: '#002952', // Darker Blue for main panels
    cardBackground: '#003C78', // Medium Blue for cards
    borderColor: '#3071A7', // Accent Blue for borders
    
    // Light Palette for Text & Accents (Warm/Sunrise)
    primaryText: '#E6E0C0', // Lightest Cream (from dark palette) for main text
    accentText: '#FFD7C5',  // Light Peach for secondary text/labels
    
    // Accents & Status (Warm Tones)
    brightAccent: '#FFECC0', // Very Light Cream/Yellow (FPS, Score, Icon)
    optimizedColor: '#E192B4', // Mid Pink for Optimized (High-Vis)
    baselineColor: '#FFD2A8', // Light Peach for Baseline
    alertColor: '#9C628C',     // Deep Mauve for Alerts (Temp, Missed, LIVE)
    
    // Derived for UI elements
    mutedIcon: '#3071A7', // Using border color for muted icons

    // New Color for Baseline Text (Dark text on light background for contrast)
    darkText: '#000D1B' // Deep Navy/Black for contrast
  };

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

  // Toggle delegate setting on Pi via proxy
  const toggleDelegate = async () => {
    try {
      const newState = !piMetrics?.delegate_enabled;
      // Optimistically update the state for a smoother UI experience
      setPiMetrics(prev => ({
          ...prev,
          delegate_enabled: newState
      }));
      await fetch(`/api/control/delegate?enable=${newState}`, {
        method: 'POST'
      });
      
      // Metrics will update automatically via polling
    } catch (error) {
      console.error('Failed to toggle delegate:', error);
      // Revert the optimistic update if the API call fails
      setPiMetrics(prev => ({
          ...prev,
          delegate_enabled: !newState
      }));
    }
  };

  // Stream configuration
  const currentStream = {
    videoUrl: '/api/stream',
    fps: piMetrics?.fps || 0,
    frameTime: piMetrics?.fps ? (1000 / piMetrics.fps).toFixed(1) : 0,
    temperature: piMetrics?.cpu_temp_c || 0,
    memoryUsage: piMetrics?.mem_used_mb ? (piMetrics.mem_used_mb / 1024).toFixed(1) : 0,
    inferenceTime: piMetrics?.infer_ms || 0,
    gameScore: piMetrics?.game_score || 0,
    gameMissed: piMetrics?.game_missed || 0,
    gameFruits: piMetrics?.game_fruits || 0,
    modelName: piMetrics?.model_name || 'Loading...',
    delegateEnabled: piMetrics?.delegate_enabled || false
  };

  const MetricCard = ({ icon: Icon, label, value, unit, color }) => (
    <div style={{
      backgroundColor: colors.cardBackground, // Medium Blue
      borderRadius: '0.375rem',
      padding: '0.5rem',
      border: `1px solid ${colors.borderColor}` // Accent Blue
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', marginBottom: '0.25rem' }}>
        <Icon style={{ width: '0.875rem', height: '0.875rem', color: color }} />
        <span style={{ color: colors.accentText, fontSize: '0.75rem' }}>{label}</span> {/* Light Peach */}
      </div>
      <div style={{ fontSize: '1rem', fontWeight: 'bold', color: colors.primaryText }}> {/* Light Cream */}
        {value}
        <span style={{ fontSize: '0.75rem', color: colors.accentText, marginLeft: '0.25rem' }}>{unit}</span> {/* Light Peach */}
      </div>
    </div>
  );

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: colors.mainBackground, // Darkest Navy
      color: colors.primaryText, // Light Cream
      padding: '0'
    }}>
      {/* Main Content */}
      <div style={{ display: 'flex', gap: '0', height: '100vh' }}>
        {/* Video Stream Section - Left Side */}
        <div style={{ flex: 1, display: 'flex' }}>
          <div style={{
            backgroundColor: colors.sectionBackground, // Darker Blue
            padding: '1rem',
            border: `1px solid ${colors.borderColor}`, // Accent Blue
            borderRight: 'none',
            width: '100%',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center', 
              justifyContent: 'space-between',
              marginBottom: '1rem' 
            }}>
              {/* Left Column: Text (Reordered and Resized) */}
              <div style={{ flex: 1 }}>
                {/* Header (Much Bigger) - NOW FIRST */}
                <h2 style={{ 
                    fontSize: '1.875rem', // text-3xl
                    fontWeight: '600', 
                    color: colors.primaryText,
                    lineHeight: '1.2' // keeps text tight
                }}>
                  Fruit Ninja AR - {currentStream.delegateEnabled ? 'ARM Optimized' : 'Baseline'}
                </h2>
                {/* Model: (NOW UNDER HEADER) */}
                <p style={{ color: colors.accentText, fontSize: '0.875rem', marginTop: '0.25rem', marginBottom: '0' }}> 
                  Model: {currentStream.modelName}
                </p>
              </div>
              
              {/* Right Column: Button and LIVE tag (Adjusted Margin) */}
              <div style={{ 
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'flex-end', 
                  gap: '0.5rem',
                  marginTop: '0.5rem' // Margin for spacing above the button group
              }}>
                <button
                  onClick={toggleDelegate}
                  disabled={loading}
                  style={{
                    backgroundColor: currentStream.delegateEnabled ? colors.optimizedColor : colors.baselineColor, // Pink vs Peach
                    marginBottom: '1.5rem', 
                    padding: '0.5rem 1rem',
                    borderRadius: '0.375rem',
                    fontWeight: '600',
                    fontSize: '0.875rem',
                    border: 'none',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    // Changed text color based on background:
                    color: currentStream.delegateEnabled ? colors.primaryText : colors.darkText, 
                    opacity: loading ? 0.7 : 1
                  }}
                  onMouseOver={(e) => {
                    if (!loading) {
                      e.target.style.backgroundColor = currentStream.delegateEnabled ? '#C07E9F' : '#E0A78F'; // Slightly darker hover
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!loading) {
                      e.target.style.backgroundColor = currentStream.delegateEnabled ? colors.optimizedColor : colors.baselineColor;
                    }
                  }}
                >
                  <Gamepad2 style={{ width: '1rem', height: '1rem' }} />
                  {loading ? 'Loading...' : (currentStream.delegateEnabled ? 'ARM Optimized' : 'Baseline')}
                </button>
                <span style={{
                  padding: '0.25rem 0.5rem',
                  backgroundColor: colors.alertColor, // Deep Mauve
                  borderRadius: '9999px',
                  fontSize: '0.75rem',
                  fontWeight: '500',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.375rem',
                  color: colors.primaryText // Light Cream text on LIVE tag
                }}>
                  <span style={{
                    width: '0.375rem',
                    height: '0.375rem',
                    backgroundColor: colors.primaryText, 
                    borderRadius: '9999px',
                    animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite'
                  }}></span>
                  LIVE
                </span>
              </div>
            </div>
            <div style={{
              flex: 1,
              backgroundColor: colors.mainBackground, // Darkest Navy
              borderRadius: '0.5rem',
              overflow: 'hidden',
              border: `1px solid ${colors.borderColor}`, // Accent Blue
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              {loading ? (
                <div style={{
                  width: '100%',
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: colors.accentText 
                }}>
                  Loading stream...
                </div>
              ) : (
                <img
                  src={currentStream.videoUrl}
                  alt="Fruit Ninja AR Stream"
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain',
                    backgroundColor: colors.mainBackground
                  }}
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextSibling.style.display = 'flex';
                  }}
                />
              )}
              <div style={{
                width: '100%',
                height: '100%',
                display: 'none',
                alignItems: 'center',
                justifyContent: 'center',
                color: colors.alertColor, // Deep Mauve
                flexDirection: 'column',
                gap: '0.5rem'
              }}>
                <p>Stream Offline</p>
                <p style={{ fontSize: '0.875rem', color: colors.accentText }}>
                  Check Pi connection
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Section - Right Side */}
        <div style={{ width: '20%', minWidth: '250px' }}>
          <div style={{
            backgroundColor: colors.sectionBackground, // Darker Blue
            padding: '1rem',
            border: `1px solid ${colors.borderColor}`, // Accent Blue
            borderLeft: 'none',
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            boxSizing: 'border-box'
          }}>
            <h2 style={{ fontSize: '1rem', fontWeight: '600', marginBottom: '0.75rem' }}>
              Performance Metrics
            </h2>
            <div 
              className="custom-scrollbar" 
              style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto' }}
            >
              <MetricCard
                icon={Activity}
                label="FPS"
                value={currentStream.fps}
                unit="fps"
                color={colors.brightAccent} 
              />
              <MetricCard
                icon={Clock}
                label="Frame Time"
                value={currentStream.frameTime}
                unit="ms"
                color={colors.baselineColor} 
              />
              <MetricCard
                icon={Clock}
                label="Inference Time"
                value={currentStream.inferenceTime}
                unit="ms"
                color={colors.baselineColor} 
              />
              <MetricCard
                icon={Thermometer}
                label="Temperature"
                value={currentStream.temperature}
                unit="°C"
                color={colors.alertColor} 
              />
              <MetricCard
                icon={HardDrive}
                label="Memory Usage"
                value={currentStream.memoryUsage}
                unit="GB"
                color={colors.mutedIcon} 
              />
              <MetricCard
                icon={Gamepad2}
                label="Game Score"
                value={currentStream.gameScore}
                unit="pts"
                color={colors.brightAccent} 
              />
              <MetricCard
                icon={Activity}
                label="Fruits Active"
                value={currentStream.gameFruits}
                unit=""
                color={colors.brightAccent} 
              />
              <MetricCard
                icon={Activity}
                label="Missed"
                value={currentStream.gameMissed}
                unit=""
                color={colors.alertColor} 
              />
            </div>
          </div>
        </div>
      </div>

      <style>{`
        /* Google Sans Emulation (Global Font Stack) */
        * {
            font-family: 'Inter', 'Roboto', 'Arial', sans-serif;
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
        
        /* Custom Scrollbar Styles */
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }

        .custom-scrollbar::-webkit-scrollbar-track {
          background: ${colors.sectionBackground}; 
          border-radius: 10px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: ${colors.borderColor}; 
          border-radius: 10px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: ${colors.accentText};
        }
      `}</style>
    </div>
  );
}