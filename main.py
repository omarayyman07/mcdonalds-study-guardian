import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import subprocess
import tempfile
import shutil
import ctypes
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 1

MODEL_PATH = "face_landmarker.task"
AUDIO_PATH = "distraction.mp3"

# Page Chrome opens when you get distracted.
WEBSITE_URL = "https://jobs.mchire.com/"

CALIBRATION_SECONDS = 3

# ============================================================
# DETECTION SENSITIVITY
# ============================================================

# Master multiplier applied to the natural head "wobble" that is
# measured during calibration. The per-axis threshold is:
#     clip(SENSITIVITY * natural_variation, MIN_*_CHANGE, MAX_*_CHANGE)
#   Lower  -> smaller movements count as DISTRACTED (twitchier)
#   Higher -> you must look further away before DISTRACTED
SENSITIVITY = 5

# Hard FLOOR for the thresholds, in degrees of head rotation away
# from your calibrated baseline. A perfectly still calibration
# cannot make the detector more sensitive than this.
#   Lower  -> more sensitive   Higher -> less sensitive
MIN_PITCH_CHANGE = 8      # looking up / down (e.g. down at a phone)
MIN_YAW_CHANGE = 10       # looking left / right

# Hard CEILING for the thresholds, in degrees. Fidgeting during
# calibration cannot make the detector impossible to trip.
#   Lower  -> stays responsive after a shaky calibration
#   Higher -> trusts the calibration spread more
MAX_PITCH_CHANGE = 22
MAX_YAW_CHANGE = 28

# ============================================================
# TEMPORAL SMOOTHING  (kills single-frame MediaPipe pose noise)
# ============================================================

# Median pre-filter window in frames (odd number).
# Removes isolated 1-frame pose spikes before the EMA sees them.
#   3 is plenty. Set 1 to disable. 5 = smoother but ~1 frame slower.
MEDIAN_WINDOW = 3

# Exponential moving average factor for pitch/yaw, range 0..1.
# THIS IS THE MAIN SMOOTHNESS KNOB.
#   Higher (0.55-0.7) -> snappier, less smooth, lower latency
#   Lower  (0.30-0.40) -> smoother, calmer, a little more latency
EMA_ALPHA = 0.45

# ============================================================
# HYSTERESIS + CONFIRMATION  (kills STUDYING<->DISTRACTED flip-flop)
# ============================================================

# Schmitt-trigger thresholds as a fraction of the calibrated
# threshold. You must exceed ENTER_RATIO * threshold to BECOME
# distracted, and drop back under EXIT_RATIO * threshold to stop.
# The gap between them is a dead-band that noise cannot cross.
#   Widen the gap (lower EXIT_RATIO)  -> more stable, slightly stickier
#   Narrow the gap (raise EXIT_RATIO) -> less sticky, more toggle risk
ENTER_RATIO = 1.00
EXIT_RATIO = 0.60

# Consecutive frames the trigger must agree before the state
# actually flips. This is a few tens of milliseconds, NOT a timer.
#   Raise -> steadier but a touch slower to react
#   Lower (min 1) -> faster but more twitchy
ENTER_FRAMES = 2
EXIT_FRAMES = 2

# ============================================================
# FACE-LOSS HANDLING
# ============================================================

# Dropouts this short (frames) just HOLD the current state and take
# no action, so a 1-2 frame flicker never toggles Chrome/audio.
FACE_LOSS_GRACE_FRAMES = 8

# If the face stays gone longer than this (frames), treat it as
# "not at the desk / turned fully away" -> DISTRACTED.
FACE_LOSS_DISTRACTED_FRAMES = 20

# ============================================================
# DEBUG
# ============================================================

# True  -> on-screen readout (pitch/yaw vs thresholds, counters, fps)
#          plus a compact console line. Toggle live with the 'd' key.
# False -> clean preview only.
DEBUG = True

CAMERA_WINDOW = "McDonald's - Study Guardian"
PREVIEW_WIDTH = 500


# ============================================================
# FIND GOOGLE CHROME
# ============================================================

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(
        r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"
    ),
]

CHROME_PATH = None

for path in CHROME_PATHS:

    if os.path.exists(path):

        CHROME_PATH = path
        break


if CHROME_PATH is None:

    print("ERROR: Google Chrome was not found.")
    print("Please install Google Chrome.")

    raise SystemExit


# ============================================================
# MEDIAPIPE
# ============================================================

print("MediaPipe version:", mp.__version__)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    output_facial_transformation_matrixes=True,
)


landmarker = FaceLandmarker.create_from_options(
    options
)


# ============================================================
# AUDIO
# ============================================================

pygame.mixer.init()


if not os.path.exists(AUDIO_PATH):

    print("ERROR: distraction.mp3 was not found.")
    print("Put distraction.mp3 in the same folder as main.py.")

    landmarker.close()
    pygame.mixer.quit()

    raise SystemExit


pygame.mixer.music.load(AUDIO_PATH)


# ============================================================
# DISTRACTION PAGE
# ============================================================

print("Distraction page:", WEBSITE_URL)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)


if not camera.isOpened():

    print("ERROR: Could not access camera.")

    landmarker.close()
    pygame.mixer.quit()

    raise SystemExit


actual_width = int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print(
    f"Camera resolution: "
    f"{actual_width} x {actual_height}"
)


# ============================================================
# CAMERA WINDOW
# ============================================================

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

aspect_ratio = actual_height / actual_width

preview_width = PREVIEW_WIDTH

preview_height = int(
    preview_width * aspect_ratio
)


cv2.namedWindow(
    CAMERA_WINDOW,
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    CAMERA_WINDOW,
    preview_width,
    preview_height
)


x = screen_width - preview_width - 20
y = 50


cv2.moveWindow(
    CAMERA_WINDOW,
    x,
    y
)


cv2.setWindowProperty(
    CAMERA_WINDOW,
    cv2.WND_PROP_TOPMOST,
    1
)


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

baseline = None
variation = None

browser_process = None
browser_profile = None

audio_playing = False

calibration_values = []

# Per-axis thresholds, filled in when calibration completes.
pitch_threshold = 0.0
yaw_threshold = 0.0

# Median pre-filter buffers (raw change-from-baseline, in degrees).
pitch_median_buf = deque(maxlen=MEDIAN_WINDOW)
yaw_median_buf = deque(maxlen=MEDIAN_WINDOW)

# EMA smoothing state (None until the first valid frame seeds it).
ema_pitch = None
ema_yaw = None

# Schmitt-trigger state machine.
is_distracted = False
enter_counter = 0
exit_counter = 0

# Face-loss tracking.
missing_frames = 0
face_present = False

# Smoothed values kept at module scope so the debug overlay can
# always read the latest numbers.
smooth_pitch = 0.0
smooth_yaw = 0.0

# Rough measured frame rate, for the debug readout only.
fps = 0.0
fps_last_tick = cv2.getTickCount()


# ============================================================
# GET HEAD ROTATION
# ============================================================

def get_head_rotation(result):

    if not result.facial_transformation_matrixes:

        return None


    matrix = result.facial_transformation_matrixes[0]

    matrix = np.array(matrix)

    rotation = matrix[:3, :3]


    sy = np.sqrt(

        rotation[0, 0] ** 2
        +
        rotation[1, 0] ** 2

    )


    if sy > 1e-6:

        pitch = np.arctan2(

            rotation[2, 1],
            rotation[2, 2]

        )

        yaw = np.arctan2(

            -rotation[2, 0],
            sy

        )

        roll = np.arctan2(

            rotation[1, 0],
            rotation[0, 0]

        )

    else:

        pitch = np.arctan2(

            -rotation[1, 2],
            rotation[1, 1]

        )

        yaw = np.arctan2(

            -rotation[2, 0],
            sy

        )

        roll = 0


    return np.degrees(
        [pitch, yaw, roll]
    )


# ============================================================
# OPEN CHROME
# ============================================================

def open_website():

    global browser_process
    global browser_profile


    if browser_process is not None:

        return


    print("Opening Chrome...")


    browser_profile = tempfile.mkdtemp(
        prefix="mcdonalds_guardian_"
    )


    browser_process = subprocess.Popen(

        [

            CHROME_PATH,

            "--new-window",

            "--no-first-run",

            "--no-default-browser-check",

            "--disable-session-crashed-bubble",

            "--disable-extensions",

            f"--user-data-dir={browser_profile}",

            WEBSITE_URL,

        ],

        stdout=subprocess.DEVNULL,

        stderr=subprocess.DEVNULL

    )


# ============================================================
# CLOSE CHROME
# ============================================================

def close_website():

    global browser_process
    global browser_profile


    if browser_process is not None:

        print("Closing Chrome...")


        try:

            subprocess.run(

                [

                    "taskkill",

                    "/PID",

                    str(browser_process.pid),

                    "/T",

                    "/F",

                ],

                stdout=subprocess.DEVNULL,

                stderr=subprocess.DEVNULL,

            )

        except Exception:

            pass


        browser_process = None


    if browser_profile is not None:

        shutil.rmtree(

            browser_profile,

            ignore_errors=True

        )

        browser_profile = None


# ============================================================
# CALIBRATION
# ============================================================

print()
print("========================================")
print("CALIBRATION")
print("========================================")
print()
print("Look normally at your computer screen.")
print("Keep your head in your normal studying position.")
print()
print(
    f"Calibrating for "
    f"{CALIBRATION_SECONDS} seconds..."
)
print()


calibration_start = cv2.getTickCount()


calibration_duration = (

    CALIBRATION_SECONDS
    *
    cv2.getTickFrequency()

)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        success, frame = camera.read()


        if not success:

            print("Could not read camera frame.")

            break


        # ====================================================
        # MEASURED FRAME RATE  (debug readout only)
        # ====================================================

        now_tick = cv2.getTickCount()

        frame_dt = (
            (now_tick - fps_last_tick)
            / cv2.getTickFrequency()
        )

        fps_last_tick = now_tick

        if frame_dt > 0:

            fps = 0.9 * fps + 0.1 * (1.0 / frame_dt)


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        rgb_frame = cv2.cvtColor(

            frame,
            cv2.COLOR_BGR2RGB

        )


        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame

        )


        # ====================================================
        # FACE DETECTION
        # ====================================================

        result = landmarker.detect_for_video(

            mp_image,
            timestamp

        )


        timestamp += 1


        # ====================================================
        # HEAD ROTATION
        # ====================================================

        rotation = get_head_rotation(result)


        # ====================================================
        # CALIBRATION
        # ====================================================

        if baseline is None:

            if rotation is not None:

                calibration_values.append(
                    rotation
                )


            elapsed = (

                cv2.getTickCount()
                -
                calibration_start

            )


            if elapsed >= calibration_duration:

                if len(calibration_values) > 10:

                    values = np.array(
                        calibration_values
                    )


                    # Robust baseline: the median ignores the odd
                    # bad frame far better than the mean.
                    baseline = np.median(
                        values,
                        axis=0
                    )


                    # Robust spread: scaled median absolute
                    # deviation (~comparable to a std dev, but a
                    # few outlier frames barely move it).
                    mad = np.median(
                        np.abs(values - baseline),
                        axis=0
                    )

                    variation = 1.4826 * mad


                    # Clamp the natural variation so neither a
                    # frozen-still nor a fidgety calibration can
                    # produce unusable thresholds.
                    variation = np.clip(
                        variation,
                        0.5,
                        3.0
                    )


                    # Pre-compute the fixed per-axis thresholds
                    # once, so the main loop stays cheap.
                    pitch_threshold = float(np.clip(
                        SENSITIVITY * variation[0],
                        MIN_PITCH_CHANGE,
                        MAX_PITCH_CHANGE
                    ))

                    yaw_threshold = float(np.clip(
                        SENSITIVITY * variation[1],
                        MIN_YAW_CHANGE,
                        MAX_YAW_CHANGE
                    ))


                    print(
                        "Calibration complete!"
                    )


                    print(
                        "Baseline:",
                        baseline
                    )


                    print(
                        "Natural variation:",
                        variation
                    )


                    print(
                        "Pitch threshold:",
                        round(pitch_threshold, 1),
                        " Yaw threshold:",
                        round(yaw_threshold, 1)
                    )


                    print()


                else:

                    print(
                        "Not enough face data."
                    )


                    print(
                        "Restarting calibration..."
                    )


                    calibration_values = []


                    calibration_start = (
                        cv2.getTickCount()
                    )


        # ====================================================
        # DEFAULT STATE
        # ====================================================

        status = "CALIBRATING..."

        direction = ""

        face_present = rotation is not None

        if face_present:

            missing_frames = 0

        else:

            missing_frames += 1

            # A dropout must never leave a half-finished
            # confirmation count lying around.
            enter_counter = 0
            exit_counter = 0


        # ====================================================
        # DETECTION  (only once calibrated, and only with a face)
        # ====================================================

        if (

            baseline is not None
            and rotation is not None

        ):

            pitch = rotation[0]

            yaw = rotation[1]

            roll = rotation[2]


            # ------------------------------------------------
            # RAW CHANGE
            # ------------------------------------------------

            pitch_change = (

                pitch
                -
                baseline[0]

            )


            yaw_change = (

                yaw
                -
                baseline[1]

            )


            # ------------------------------------------------
            # 1) MEDIAN PRE-FILTER  (drop lone 1-frame spikes)
            # ------------------------------------------------

            pitch_median_buf.append(pitch_change)
            yaw_median_buf.append(yaw_change)

            med_pitch = float(np.median(pitch_median_buf))
            med_yaw = float(np.median(yaw_median_buf))


            # ------------------------------------------------
            # 2) EMA  (the actual smoothing)
            # ------------------------------------------------

            if ema_pitch is None:

                ema_pitch = med_pitch
                ema_yaw = med_yaw

            else:

                ema_pitch = (
                    EMA_ALPHA * med_pitch
                    + (1.0 - EMA_ALPHA) * ema_pitch
                )

                ema_yaw = (
                    EMA_ALPHA * med_yaw
                    + (1.0 - EMA_ALPHA) * ema_yaw
                )

            smooth_pitch = ema_pitch
            smooth_yaw = ema_yaw


            # ------------------------------------------------
            # 3) SCHMITT TRIGGER  (separate enter / exit bars)
            # ------------------------------------------------

            over_enter = (

                abs(smooth_pitch) > pitch_threshold * ENTER_RATIO

                or

                abs(smooth_yaw) > yaw_threshold * ENTER_RATIO

            )

            over_exit = (

                abs(smooth_pitch) > pitch_threshold * EXIT_RATIO

                or

                abs(smooth_yaw) > yaw_threshold * EXIT_RATIO

            )


            # ------------------------------------------------
            # 4) CONFIRMATION  (N consecutive agreeing frames)
            # ------------------------------------------------

            if not is_distracted:

                enter_counter = (
                    enter_counter + 1 if over_enter else 0
                )

                exit_counter = 0

                if enter_counter >= ENTER_FRAMES:

                    is_distracted = True
                    enter_counter = 0

            else:

                exit_counter = (
                    exit_counter + 1 if not over_exit else 0
                )

                enter_counter = 0

                if exit_counter >= EXIT_FRAMES:

                    is_distracted = False
                    exit_counter = 0


            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            if smooth_yaw > yaw_threshold:

                direction = "LOOKING RIGHT"

            elif smooth_yaw < -yaw_threshold:

                direction = "LOOKING LEFT"

            elif smooth_pitch > pitch_threshold:

                direction = "LOOKING DOWN"

            elif smooth_pitch < -pitch_threshold:

                direction = "LOOKING UP"

            else:

                direction = "FACING SCREEN"


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if is_distracted:

                status = "DISTRACTED"

            else:

                status = "STUDYING"


        # ====================================================
        # FACE-LOSS HANDLING  (calibrated, but no face this frame)
        # ====================================================

        elif baseline is not None:

            if missing_frames <= FACE_LOSS_GRACE_FRAMES:

                # Just a flicker: hold the current state, no action.
                status = (
                    "DISTRACTED" if is_distracted else "STUDYING"
                )

                direction = "FACE LOST (brief)"

            elif missing_frames >= FACE_LOSS_DISTRACTED_FRAMES:

                # Genuinely gone: you are not at the screen studying.
                is_distracted = True

                status = "DISTRACTED"

                direction = "NO FACE"

            else:

                # Between the two: keep holding the last known state.
                status = (
                    "DISTRACTED" if is_distracted else "STUDYING"
                )

                direction = "FACE LOST"


            # Anything longer than a flicker: wipe the smoothing so
            # re-acquiring the face starts from a clean value rather
            # than replaying a stale one.
            if missing_frames > FACE_LOSS_GRACE_FRAMES:

                pitch_median_buf.clear()
                yaw_median_buf.clear()

                ema_pitch = None
                ema_yaw = None

                smooth_pitch = 0.0
                smooth_yaw = 0.0


        # ====================================================
        # DISTRACTION ACTION
        # ====================================================

        if status == "DISTRACTED":

            # ------------------------------------------------
            # OPEN CHROME
            # ------------------------------------------------

            if (

                browser_process is None

                or

                browser_process.poll() is not None

            ):

                browser_process = None

                open_website()


            # ------------------------------------------------
            # PLAY AUDIO
            # ------------------------------------------------

            if not audio_playing:

                print("DISTRACTED!")

                pygame.mixer.music.play(-1)

                audio_playing = True


        # ====================================================
        # STUDYING ACTION
        # ====================================================

        elif status == "STUDYING":

            # ------------------------------------------------
            # STOP AUDIO
            # ------------------------------------------------

            if audio_playing:

                pygame.mixer.music.stop()

                audio_playing = False


            # ------------------------------------------------
            # CLOSE CHROME
            # ------------------------------------------------

            if browser_process is not None:

                close_website()


        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        # Colour-code the headline so state is readable at a glance.
        if status == "STUDYING":

            status_color = (0, 200, 0)

        elif status == "DISTRACTED":

            status_color = (0, 0, 255)

        else:

            status_color = (0, 200, 255)


        cv2.putText(

            frame,

            status,

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            status_color,

            2

        )


        # ====================================================
        # DISPLAY DIRECTION
        # ====================================================

        if direction:

            cv2.putText(

                frame,

                direction,

                (20, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2

            )


        # ====================================================
        # DEBUG OVERLAY  (toggle with the 'd' key)
        # ====================================================

        if DEBUG:

            debug_lines = [
                f"face: {'YES' if face_present else 'NO'}"
                f"  missing: {missing_frames}",

                f"pitch  {smooth_pitch:+6.1f} "
                f"/ enter {pitch_threshold * ENTER_RATIO:4.1f} "
                f"exit {pitch_threshold * EXIT_RATIO:4.1f}",

                f"yaw    {smooth_yaw:+6.1f} "
                f"/ enter {yaw_threshold * ENTER_RATIO:4.1f} "
                f"exit {yaw_threshold * EXIT_RATIO:4.1f}",

                f"enter_cnt {enter_counter}  "
                f"exit_cnt {exit_counter}  "
                f"fps {fps:4.1f}",
            ]

            for i, line in enumerate(debug_lines):

                cv2.putText(
                    frame,
                    line,
                    (20, 120 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # Compact console line ~4x/second so you can tune
            # without staring at the preview.
            if timestamp % 5 == 0:

                print(
                    f"[{status:11s}] "
                    f"face={'Y' if face_present else 'N'} "
                    f"miss={missing_frames:2d} "
                    f"p={smooth_pitch:+6.1f} "
                    f"y={smooth_yaw:+6.1f} "
                    f"pth={pitch_threshold:4.1f} "
                    f"yth={yaw_threshold:4.1f} "
                    f"ec={enter_counter} xc={exit_counter} "
                    f"fps={fps:4.1f}"
                )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            CAMERA_WINDOW,

            frame

        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        # Toggle the debug overlay live.
        if key == ord("d"):

            DEBUG = not DEBUG


# ============================================================
# CLEANUP
# ============================================================

finally:

    print()
    print("Shutting down...")


    if audio_playing:

        pygame.mixer.music.stop()

        audio_playing = False


    close_website()


    camera.release()


    landmarker.close()


    cv2.destroyAllWindows()


    pygame.mixer.quit()


    print(
        "McDonald's Study Guardian stopped."
    )