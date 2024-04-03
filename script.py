import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyautogui
import numpy as np
import time

model_path = "face_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

key = 0
screen_size = pyautogui.size()
screen_size = [size - 2 for size in screen_size]
# pos = list(pyautogui.position())
cals = [
    None,
    None,
    None,
    None
]
M = np.zeros((3, 3))
speed = [0, 0]
mouse_down = [False, False]
last_movement = [time.time()]

# Create a face landmarker instance with the live stream mode:
def print_result(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    if len(result.face_blendshapes) != 1:
        return

    if result.face_blendshapes[0][1].score < 1e-3:
        if not mouse_down[0]:
            mouse_down[0] = True
            pyautogui.mouseDown(button="left")
    elif mouse_down[0]:
        mouse_down[0] = False
        pyautogui.mouseUp(button="left")

    if result.face_blendshapes[0][2].score < 1e-3:
        if not mouse_down[1]:
            mouse_down[1] = True
            pyautogui.mouseDown(button="right")
    elif mouse_down[1]:
        mouse_down[1] = False
        pyautogui.mouseUp(button="right")

    d_y = result.face_blendshapes[0][11].score # dl
    d_y += result.face_blendshapes[0][12].score # dr
    d_y -= result.face_blendshapes[0][17].score # ul
    d_y -= result.face_blendshapes[0][18].score # ur

    d_x = result.face_blendshapes[0][13].score # il
    d_x -= result.face_blendshapes[0][14].score # ir
    d_x -= result.face_blendshapes[0][15].score # ol
    d_x += result.face_blendshapes[0][16].score # or


    if (time.time() - last_movement[0]) > 0.09:
        moved = False
        delta_x, delta_y = 0, 0
        if abs(d_x) > 1:
            speed[0] = min(speed[0] + 4, 20)
            delta_x = int(np.sign(d_x)) * speed[0]
            moved = True
        else:
            speed[0] = 0
        if d_y > 1 or d_y < -0.7:
            speed[1] = min(speed[1] + 4, 20)
            delta_y += int(np.sign(d_y)) * speed[1]
            moved = True
        else:
            speed[1] = 0
        # print((pos_x, pos_y))
        if moved:
            pyautogui.move(delta_x, delta_y, duration = 0.09)
        last_movement[0] = time.time()


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_face_blendshapes=True,
    num_faces=1
)


vid = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    # The landmarker is initialized. Use it here.
    # ...
    while (True):
        frame_exists, frame = vid.read()

        # cv2.imshow("frame", frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #     break

        if frame_exists:
            frame_timestamp_ms = vid.get(cv2.CAP_PROP_POS_MSEC)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, int(frame_timestamp_ms))

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
