from gpiozero import LED, Button
from picamera2 import Picamera2
import face_recognition
import cv2
import numpy as np
import pickle
import time
import sys
import os

GREEN_PIN = 17
RED_PIN = 27
BUTTON_PIN = 22

PIN_OK = "1234"
EMERGENCY_PIN = "0000"

MAX_PIN_TRIES = 3
LOCKOUT_SECONDS = 15

COOLDOWN_SECONDS = 3
FACE_RETRIES_BEFORE_PIN = 2

THRESHOLD = 0.5
CV_SCALE = 0.25

authorized_names = ["baher", "seif", "hamza"]
admin_names = ["baher"]

LOG_FILE = "access_log.csv"

green = LED(GREEN_PIN)
red = LED(RED_PIN)
button = Button(BUTTON_PIN, pull_up=True)

with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_encodings = data["encodings"]
known_names = data["names"]

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "XRGB8888", "size": (1280, 720)}))
picam2.start()
time.sleep(0.3)

last_attempt_time = 0
pin_tries = 0
silent_mode = False

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log_event(event, who="", dist=""):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if new_file:
            f.write("time,event,who,dist\n")
        f.write(f"{ts()},{event},{who},{dist}\n")

def banner(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def led_on(led):
    if not silent_mode:
        led.on()

def led_off(led):
    led.off()

def blink(led, seconds, on_s=0.12, off_s=0.12):
    if silent_mode:
        return
    end_t = time.time() + seconds
    while time.time() < end_t:
        led.on()
        time.sleep(on_s)
        led.off()
        time.sleep(off_s)

def unlock_green(seconds=5):
    led_off(red)
    led_on(green)
    time.sleep(seconds)
    led_off(green)

def flash_red(seconds=3):
    blink(red, seconds, 0.12, 0.12)
    led_off(red)

def lockout(seconds):
    banner(f"LOCKOUT {seconds}s")
    log_event("lockout")
    end_t = time.time() + seconds
    while time.time() < end_t:
        blink(red, 0.5, 0.08, 0.08)
        time.sleep(0.2)
    led_off(red)

def capture_rgb_scaled():
    frame = picam2.capture_array()
    small = cv2.resize(frame, (0, 0), fx=CV_SCALE, fy=CV_SCALE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return rgb

def check_face_once():
    rgb = capture_rgb_scaled()
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)

    if len(encs) == 0:
        return False, "NoFace", None, "no_face"

    best_name = "Unknown"
    best_dist = None

    for enc in encs:
        dists = face_recognition.face_distance(known_encodings, enc)
        best = int(np.argmin(dists))
        name = known_names[best]
        dist = float(dists[best])

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_name = name

        if dist < THRESHOLD and name in authorized_names:
            return True, name, dist, "authorized"

        if dist < THRESHOLD and name in admin_names:
            return True, name, dist, "admin"

    return False, best_name, best_dist, "unknown"

def check_face_with_retries():
    for i in range(FACE_RETRIES_BEFORE_PIN):
        ok, who, dist, reason = check_face_once()
        if ok:
            return ok, who, dist, reason
        banner(f"Face retry {i+1}/{FACE_RETRIES_BEFORE_PIN}")
        blink(green, 0.2, 0.06, 0.06)
        time.sleep(0.2)
    return False, who, dist, reason

def wait_for_button_toggle_silent():
    global silent_mode
    banner("READY  Press button (short)  Long press=Silent toggle")
    button.wait_for_press()
    t0 = time.time()
    button.wait_for_release()
    held = time.time() - t0
    if held >= 1.2:
        silent_mode = not silent_mode
        banner(f"SILENT MODE {'ON' if silent_mode else 'OFF'}")
        log_event("silent_toggle", "on" if silent_mode else "off")
        blink(red, 0.4, 0.08, 0.08)
        return False
    return True

def ask_pin():
    try:
        return input("Enter 4-digit PIN: ").strip()
    except KeyboardInterrupt:
        banner("Exit")
        sys.exit(0)

banner("SYSTEM START")
banner(f"Authorized: {authorized_names}")
banner(f"Admin: {admin_names}")
log_event("start")

while True:
    led_off(green)
    led_off(red)

    now = time.time()
    if now - last_attempt_time < COOLDOWN_SECONDS:
        time.sleep(0.05)
        continue

    if not wait_for_button_toggle_silent():
        last_attempt_time = time.time()
        continue

    last_attempt_time = time.time()

    banner("CAPTURE  Checking face")
    blink(green, 0.25, 0.06, 0.06)

    ok, who, dist, reason = check_face_with_retries()

    if ok:
        banner(f"GRANTED  {who}  dist={dist:.3f}")
        log_event("granted_face", who, f"{dist:.3f}")
        pin_tries = 0
        unlock_green(5)
        continue

    if reason == "no_face":
        banner("DENIED  No face detected")
        log_event("denied_no_face")
    else:
        if dist is None:
            banner(f"DENIED  best={who}")
            log_event("denied_unknown", who, "")
        else:
            banner(f"DENIED  best={who}  dist={dist:.3f}")
            log_event("denied_face", who, f"{dist:.3f}")

    led_on(red)

    while True:
        pin = ask_pin()

        if pin == EMERGENCY_PIN:
            banner("EMERGENCY PIN  Unlock")
            log_event("emergency_pin")
            pin_tries = 0
            unlock_green(5)
            break

        if pin == PIN_OK:
            banner("PIN OK  Unlock")
            log_event("granted_pin")
            pin_tries = 0
            unlock_green(5)
            break

        pin_tries += 1
        banner(f"PIN BAD  tries={pin_tries}/{MAX_PIN_TRIES}")
        log_event("bad_pin", "", str(pin_tries))
        blink(red, 1.2, 0.12, 0.12)
        led_on(red)

        if pin_tries >= MAX_PIN_TRIES:
            led_off(red)
            lockout(LOCKOUT_SECONDS)
            pin_tries = 0
            break
