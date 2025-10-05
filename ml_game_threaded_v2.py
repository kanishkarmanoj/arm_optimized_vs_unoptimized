import os, time, threading, json, cv2, numpy as np, requests, subprocess
from flask import Flask, Response

cv2.setUseOptimized(True); cv2.setNumThreads(4)

DELEG = os.path.expanduser("~/ctl_srv/delegate.json")
MET   = "http://127.0.0.1:8082/metrics/update"
MODEL = os.path.expanduser("~/models/hand.tflite")

tfl=None; load_delegate=None
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    tfl=Interpreter
except Exception: pass

app=Flask(__name__); latest=None; lock=threading.Lock()

# Threaded camera capture globals with frame versioning
latest_camera_frame = None
latest_frame_id = 0
camera_frame_lock = threading.Lock()
camera_running = True

def jpg(f): ok,b=cv2.imencode(".jpg", f,[int(cv2.IMWRITE_JPEG_QUALITY),60]); return b.tobytes() if ok else None

@app.route("/stream.mjpg")
def s():
    def g():
        global latest
        while True:
            with lock: b=latest
            if b: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+b+b"\r\n"
            time.sleep(0.01)
    return Response(g(), mimetype="multipart/x-mixed-replace; boundary=frame")

def on():
    try: return json.load(open(DELEG)).get("enable",False)
    except: return False

def make_interpreter(use_delegate):
    if not tfl or not os.path.exists(MODEL): return None, None
    if use_delegate:
        os.environ.pop("TF_LITE_DISABLE_XNNPACK", None)
        threads = 4
    else:
        os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
        threads = 1

    delegates=[]
    if use_delegate and load_delegate:
        for p in ["/opt/armnn/build/delegate/libarmnnDelegate.so","libarmnnDelegate.so"]:
            try: delegates=[load_delegate(p,{})]; break
            except Exception: pass

    try:
        interp = tfl(model_path=MODEL, experimental_delegates=delegates, num_threads=threads)
    except TypeError:
        interp = tfl(model_path=MODEL)

    try: interp.allocate_tensors()
    except Exception: return None, None

    idets = interp.get_input_details()[0]
    odets = interp.get_output_details()[0]
    ishape = idets["shape"]
    return interp, (idets, odets, ishape, delegates)

def run_infer(interp, io_meta, frame):
    if not interp or not io_meta: return 0.0, None
    idets, odets, ishape, _ = io_meta
    h, w = int(ishape[1]), int(ishape[2])
    x = cv2.resize(frame, (w, h))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if idets["dtype"] == np.uint8:
        pass
    else:
        x = (x.astype(np.float32) - 127.5) / 127.5
    x = np.expand_dims(x, 0).astype(idets["dtype"])
    interp.set_tensor(idets["index"], x)
    t0=time.time(); interp.invoke(); infer_ms=(time.time()-t0)*1000.0
    try:
        y = interp.get_tensor(odets["index"])
    except Exception:
        y = None
    return infer_ms, y

def camera_thread():
    """Dedicated camera capture thread with frame versioning"""
    global latest_camera_frame, latest_frame_id, camera_running

    try:
        subprocess.run(["v4l2-ctl", "--device=/dev/video0",
                       "--set-ctrl=exposure_dynamic_framerate=0"],
                      stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(3, 640); cap.set(4, 360); cap.set(5, 30)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = ''.join([chr((fourcc >> (8*i)) & 0xFF) for i in range(4)])
    print(f"Camera thread: {fcc} @ {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5)} fps")

    while camera_running:
        ret, frame = cap.read()
        if ret:
            with camera_frame_lock:
                latest_camera_frame = frame
                latest_frame_id += 1

    cap.release()
    print("Camera thread stopped")

def loop():
    global latest_camera_frame, latest_frame_id

    interp=None; meta=None; accel=False; last=0
    last_processed_id = -1

    while True:
        t0=time.time()

        # Wait for a NEW frame (frame ID changed)
        while True:
            with camera_frame_lock:
                current_id = latest_frame_id
                if current_id != last_processed_id and latest_camera_frame is not None:
                    fr = latest_camera_frame.copy()
                    last_processed_id = current_id
                    break
            time.sleep(0.001)  # Brief sleep while waiting for new frame

        want=on()
        if want!=accel:
            accel=want
            interp, meta = make_interpreter(accel)

        # HSV blade overlay
        hsv=cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(40,40,40),(80,255,255))
        cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            (x,y),r=cv2.minEnclosingCircle(max(cnts,key=cv2.contourArea))
            cv2.circle(fr,(int(x),int(y)),int(r),(0,255,0),2)

        infer_ms=0.0
        if interp:
            infer_ms,_ = run_infer(interp, meta, fr)

        fps = 1.0 / max(1e-6, time.time()-t0)
        using_armnn = bool(meta and meta[-1])
        label = ("TFLite+ArmNN" if (accel and using_armnn)
                 else "TFLite (XNN ON)" if accel
                 else "TFLite (XNN OFF)" if interp
                 else "HSV")
        cv2.putText(fr, f"{label} | fps {fps:4.1f} | infer {infer_ms:4.1f}ms",
                    (12,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        b=jpg(fr)
        if b:
            global latest
            with lock: latest=b

        if time.time()-last>0.3:
            try:
                mem=int(os.popen("free -m | awk '/Mem:/{print $3}'").read() or 0)
            except: mem=0
            try:
                requests.post(MET, json={
                    "fps":round(fps,1),"infer_ms":round(infer_ms,1),
                    "pre_ms":0.0,"post_ms":1.0,"mem_used_mb":mem,
                    "delegated_ops":int(using_armnn),"neon_build":True,"model_name":label
                }, timeout=0.4)
            except: pass
            last=time.time()

def run():
    cam_thread = threading.Thread(target=camera_thread, daemon=True, name="CameraCapture")
    cam_thread.start()
    time.sleep(0.3)
    threading.Thread(target=loop,daemon=True).start()
    app.run(host="0.0.0.0", port=8090, threaded=True)

if __name__=="__main__":
    run()
