import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors

for m in get_monitors():
    print(str(m))


x_cam = 0
y_cam = 0


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(
            f"Mouse down at (x_screen: {x}, y_screen: {y}), (x_cam: {x_cam}, y_cam: {y_cam})"
        )


# import pyautogui
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# screen_w, screen_h = pyautogui.size()

# Set screen width and height
screen_width = 1920
screen_height = 1080
# screen_width = get_monitors()[2].width
# screen_height = get_monitors()[2].height
distance = 50

print(screen_width, screen_height)

df = pd.read_csv("landmarks_2.csv", header=None)

# convert column values of x and y to float after dropping first row
df = df.drop(df.index[0])
df = df.astype(float)
## get min and max values of x and y of df
x_min = df[0].min()
x_max = df[0].max()
y_min = df[1].min()
y_max = df[1].max()


## generate a scale calculator function to scale the values between x_min and x_max to 0 and 1 and y_min and y_max to 0 and 1
def scale_calculator(n, n_min, n_max):
    if n <= n_min:
        return 0
    elif n >= n_max:
        return 1
    return (n - n_min) / (n_max - n_min)


# Set the distance between dots

# Initialize an empty black image
image = np.zeros((screen_height, screen_width, 3), np.uint8)
screenShapeW, screenShapeH = image.shape[:2]

# Calculate the number of dots per column and the horizontal spacing between columns
num_dots = 3
col_spacing = screen_width // 3

# Calculate the horizontal offset for the dots
offset_x = (screen_width - (col_spacing * (num_dots - 1))) // 4
# Calculate the vertical spacing between dots
dot_spacing = screen_height // (num_dots)

positions_x = [50, screen_width // 2, screen_width - 50]
positions_y = [50, screen_height // 2, screen_height - 50]


for col in positions_x:
    for dot in positions_y:
        if col == 50 and dot == 50:
            cv2.circle(image, (400, dot), 5, (255, 0, 255), -1)
        else:
            cv2.circle(image, (col, dot), 5, (255, 0, 255), -1)

cols = ["x", "y"]
df = pd.DataFrame(columns=cols)

while True:
    cv2.namedWindow("EyeGaze", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("EyeGaze", mouse_callback)
    _, frame = cam.read()
    frame = cv2.addWeighted(frame, 0.3, np.zeros(frame.shape, frame.dtype), 0, 0)
    frame = cv2.flip(frame, 1)
    start_point = (98, 48)
    end_point = (283, 258)
    cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:

        landmarks = landmark_points[0].landmark
        avg_x = sum(landmark.x for landmark in landmarks[474:478]) / 4
        avg_y = sum(landmark.y for landmark in landmarks[474:478]) / 4

        new_row = {"x": int(avg_x * frame_w), "y": int(avg_y * frame_h)}
        df.loc[-1] = new_row
        new_df = pd.DataFrame([new_row], columns=cols)
        df = pd.concat([df, new_df], ignore_index=True)

        x_cam = float(avg_x)
        y_cam = float(avg_y)
        print(
            int(
                float(scale_calculator(int(avg_x * frame_w), x_min, x_max))
                * screen_width
            ),
            int(
                float(scale_calculator(int(avg_y * frame_h), y_min, y_max))
                * screen_height
            ),
        )
        cv2.circle(frame, (int(avg_x * frame_w), int(avg_y * frame_h)), 3, (0, 255, 0))
        cv2.circle(
            image,
            (
                int(
                    float(scale_calculator(int(avg_x * frame_w), x_min, x_max))
                    * screen_width
                ),
                int(
                    float(scale_calculator(int(avg_y * frame_h), y_min, y_max))
                    * screen_height
                ),
            ),
            3,
            (0, 255, 0),
        )

    # df.to_csv("landmarks.csv", index=False)

    x_offset = y_offset = 10
    image[
        y_offset : y_offset + frame.shape[0], x_offset : x_offset + frame.shape[1]
    ] = frame

    cv2.imshow("EyeGaze", image)
    cv2.waitKey(1)
