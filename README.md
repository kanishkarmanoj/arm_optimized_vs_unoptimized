# Fruit Ninja AR Demo - Team Handoff Guide

## 📁 Project Structure (CLEAN VERSION)

```
arm_optimized_vs_unoptimized/
├── fruit_ninja_pi_optimized.py    ⭐ FINAL PRODUCTION CODE (runs on Pi)
├── claude.md                       📖 Complete project documentation
├── diagnose_camera.py              🔧 Camera diagnostic tool (optional)
├── fruit_ninja_game/               📦 Original baseline/optimized versions (reference only)
│   ├── fruit_ninja_baseline.py
│   └── fruit_ninja_optimized.py
└── Webpage/                        🌐 Frontend dashboard (your work!)
```

---

## ⭐ THE MAIN FILE: `fruit_ninja_pi_optimized.py`

**This is the ONLY file running on the Raspberry Pi.**

### What It Does:
- Real-time AR Fruit Ninja game with hand tracking
- Toggleable modes: HSV (baseline) vs MediaPipe (optimized)
- MJPEG streaming @ `http://100.81.50.51:8090/stream.mjpg`
- Metrics API @ `http://100.81.50.51:8082/metrics`
- Runs as systemd service: `ab-ml.service`

### Performance:
- **Baseline (HSV):** 30 FPS locked, simple color detection
- **Optimized (MediaPipe):** 28 FPS effective, ML-based hand tracking

---

## 🚀 Quick Start for Demo

### 1. View the Stream
```bash
# On any browser
open http://100.81.50.51:8090/stream.mjpg
```

### 2. Toggle Modes
```bash
# Switch to MediaPipe (Optimized)
curl -X POST "http://100.81.50.51:8082/control/delegate?enable=true"

# Switch to HSV (Baseline)
curl -X POST "http://100.81.50.51:8082/control/delegate?enable=false"
```

### 3. Get Metrics
```bash
curl -s http://100.81.50.51:8082/metrics | python3 -m json.tool
```

### 4. Restart Game
```bash
ssh prajit@100.81.50.51 "sudo systemctl restart ab-ml.service"
```

---

## 📂 Files on Raspberry Pi

**Location:** `/home/prajit/ab_demo/`

```
ab_demo/
├── fruit_ninja_pi_optimized.py    ⭐ Main game (this file is auto-started)
├── ctl_srv.py                      🎛️ Metrics server (runs as ab-ctl.service)
├── diagnose_camera.py              🔧 Camera testing tool
└── fruit_ninja_game/               📦 Reference code (not used in production)
```

**Virtual Environment:** `/home/prajit/.venvs/abenv311`

---

## 🎮 How The Game Works

### Architecture (3 Threads)
1. **Camera Thread:** Captures frames @ 30 FPS from USB camera
2. **Game Loop Thread:** Hand tracking + physics + rendering @ 30 FPS
3. **JPEG Encoder Thread:** Compresses frames for streaming

### Hand Tracking Modes

#### Baseline (HSV Color Detection)
- Simple color thresholding in HSV space
- **Fast:** 1.5ms inference, 272 FPS capable
- **Inaccurate:** Lighting dependent, false positives
- **Purpose:** Show "before optimization"

#### Optimized (MediaPipe Hands)
- ML-based landmark detection (21 points per hand)
- Tracks index fingertip (landmark 8) for slicing
- **Accurate:** Works in varied lighting
- **Smart:** Frame skipping (every 4th frame) + interpolation
- **Performance:** 28 FPS effective, 109ms inference
- **Purpose:** Show "ARM optimization" with ML

### Game Mechanics
- Fruits spawn from bottom, arc upward with physics
- Hand motion > 15 px/frame triggers slice
- Score +10 per slice, track missed fruits
- Particle effects on slice (12 particles)
- Motion trail follows hand (15 points)

---

## 🌐 Network Access

### Current Setup (Tailscale)
- **SSH:** `ssh prajit@100.81.50.51`
- **Stream:** `http://100.81.50.51:8090/stream.mjpg`
- **Metrics:** `http://100.81.50.51:8082/metrics`

### Demo Day Setup (Pi as WiFi AP)
- **Pi IP:** `10.42.0.1`
- **Stream:** `http://10.42.0.1:8090/stream.mjpg`
- **Latency:** <5ms (direct WiFi)
- **Note:** No internet access, cannot SSH

---

## 🛠️ Making Changes

### Edit Code Locally
1. Edit `fruit_ninja_pi_optimized.py` on your Mac
2. Sync to Pi:
   ```bash
   scp fruit_ninja_pi_optimized.py prajit@100.81.50.51:/home/prajit/ab_demo/
   ```
3. Restart service:
   ```bash
   ssh prajit@100.81.50.51 "sudo systemctl restart ab-ml.service"
   ```

### Important Files to NOT Touch
- `ctl_srv.py` - Metrics server (working)
- Service files in `/etc/systemd/system/` (already configured)

---

## 📊 Metrics API Response

```json
{
  "fps": 28.5,
  "infer_ms": 109.3,
  "cpu_temp_c": 80.3,
  "cpu_freq_mhz": 1800,
  "mem_used_mb": 982,
  "game_score": 40,
  "game_missed": 13,
  "game_fruits": 1,
  "model_name": "ARM Optimized (MediaPipe Hands)",
  "delegated_ops": 1,
  "neon_build": true
}
```

Use this data for your dashboard/visualization!

---

## 🔧 Troubleshooting

### Stream not loading?
```bash
# Check if service is running
ssh prajit@100.81.50.51 "systemctl status ab-ml.service"

# View logs
ssh prajit@100.81.50.51 "sudo journalctl -u ab-ml.service -f"
```

### Game running slow?
- Check CPU temp: Should be < 85°C
- Restart service: `sudo systemctl restart ab-ml.service`
- Verify camera: `v4l2-ctl --device=/dev/video0 --all`

### Want to test camera?
```bash
ssh prajit@100.81.50.51
cd /home/prajit/ab_demo
source /home/prajit/.venvs/abenv311/bin/activate
python diagnose_camera.py
```

---

## 📚 Full Documentation

See `claude.md` for complete project context including:
- Optimization journey (how we got from 6 FPS → 28 FPS)
- Camera configuration details
- Threading architecture
- Performance benchmarks
- Known issues & workarounds

---

## 👥 Team Responsibilities

### Backend (Prajit) ✅ COMPLETE
- [x] Hand tracking implementation (HSV + MediaPipe)
- [x] Performance optimization (threading, frame skipping)
- [x] Streaming server (MJPEG @ 30 FPS)
- [x] Metrics API
- [x] Systemd services setup
- [x] Code cleanup & sync

### Frontend (Your Team) 🚧 IN PROGRESS
- [ ] Game visual polish (UI, effects, styling)
- [ ] Web dashboard for metrics
- [ ] Toggle UI (instead of curl commands)
- [ ] Demo day presentation materials

---

## 🎯 Demo Day Checklist

1. **Before Demo:**
   - [ ] Switch to Pi AP mode (10.42.0.1)
   - [ ] Verify stream is smooth (<5ms latency)
   - [ ] Test both modes (Baseline + Optimized)
   - [ ] Prepare metrics dashboard

2. **During Demo:**
   - [ ] Start in Baseline mode (show it's fast but inaccurate)
   - [ ] Toggle to Optimized mode (show ML accuracy)
   - [ ] Let judge play with hand gestures
   - [ ] Show live metrics on dashboard

3. **Backup Plan:**
   - [ ] Have video recording ready
   - [ ] Screenshots of metrics
   - [ ] Presentation slides explaining architecture

---

## 📞 Contact

**Issues/Questions:** Message Prajit
**Git Repo:** `https://github.com/kanishkarmanoj/arm_optimized_vs_unoptimized`
**Branch:** `main`

**Last Updated:** October 5, 2024
**Status:** ✅ Production Ready
