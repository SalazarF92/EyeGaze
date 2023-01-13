import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
from os.path import exists
from csv import writer


for m in get_monitors():
    print(str(m))
## define global df_read as a dataframe

cols = ["x", "y"]
df_read = pd.DataFrame(columns=cols)


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)


# Set screen width and height
screen_width = 1920
screen_height = 1080
# screen_width = get_monitors()[2].width
# screen_height = get_monitors()[2].height

## generate a scale calculator function to scale the values between x_min and x_max to 0 and 1 and y_min and y_max to 0 and 1
def scale_calculator(n, n_min, n_max):
    if n <= n_min:
        return 0
    elif n >= n_max:
        return 1
    return (n - n_min) / (n_max - n_min)


# Initialize an empty black image
image = np.zeros((screen_height, screen_width, 3), np.uint8)
screenShapeW, screenShapeH = image.shape[:2]

positions_x = [50, screen_width // 2, screen_width - 50]
positions_y = [50, screen_height // 2, screen_height - 50]

# Set the distance between dots
for col in positions_x:
    for dot in positions_y:
        if col == 50 and dot == 50:
            cv2.circle(image, (400, dot), 5, (255, 0, 255), -1)
        else:
            cv2.circle(image, (col, dot), 5, (255, 0, 255), -1)

button_x = 450
button_y = 450
button_width = 350
button_height = 150


def on_button_clicked():
    print("Button clicked!")


def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print("outsiders")
        df = pd.DataFrame(columns=cols)
        ## write a row into df 
        if exists("landmarks.csv"):
            with open("landmarks.csv", "a") as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(param)
                f_object.close()
        else:
            df = df.append(pd.Series(param, index=cols), ignore_index=True)
            df.to_csv("landmarks.csv", index=False)


while cam.isOpened():
    cv2.namedWindow("EyeGaze", cv2.WINDOW_NORMAL)
    cv2.rectangle(
        image,
        (button_x, button_y),
        (button_x + button_width, button_y + button_height),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        image,
        "Click me!",
        (button_x + 10, button_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    _, frame = cam.read()
    frame = cv2.addWeighted(frame, 0.3, np.zeros(frame.shape, frame.dtype), 0, 0)
    frame = cv2.flip(frame, 1)

    start_point_rec_frame = (98, 48)
    end_point_rec_frame = (283, 258)
    cv2.rectangle(frame, start_point_rec_frame, end_point_rec_frame, (0, 255, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:

        landmarks = landmark_points[0].landmark
        avg_x = sum(landmark.x for landmark in landmarks[474:478]) / 4
        avg_y = sum(landmark.y for landmark in landmarks[474:478]) / 4

        def eyes():
            print("chamei eyes")
            ## open the csv file and read it
            df = pd.read_csv("landmarks.csv", header=None)
            # convert column values of x and y to float after dropping first row
            df = df.drop(df.index[0])
            df = df.astype(float)
            ## get min and max values of x and y of df
            x_min = df[0].min()
            x_max = df[0].max()
            y_min = df[1].min()
            y_max = df[1].max()
            
            if len(df) > 8:
                while True:
                    cv2.namedWindow("EyeGaze", cv2.WINDOW_NORMAL)
                    _, frame = cam.read()
                    frame = cv2.addWeighted(frame, 0.3, np.zeros(frame.shape, frame.dtype), 0, 0)
                    frame = cv2.flip(frame, 1)
                
                    start_point_rec_frame = (98, 48)
                    end_point_rec_frame = (283, 258)
                    cv2.rectangle(frame, start_point_rec_frame, end_point_rec_frame, (0, 255, 0), 2)
                
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    output = face_mesh.process(rgb_frame)
                    landmark_points = output.multi_face_landmarks
                    frame_h, frame_w, _ = frame.shape
                    if landmark_points:
                    
                        landmarks = landmark_points[0].landmark
                        avg_x = sum(landmark.x for landmark in landmarks[474:478]) / 4
                        avg_y = sum(landmark.y for landmark in landmarks[474:478]) / 4

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
                    
                    cv2.circle(frame, (int(avg_x * frame_w), int(avg_y * frame_h)), 3, (0, 255, 0))

                    x_offset = y_offset = 10
                    image[
                        y_offset : y_offset + frame.shape[0], x_offset : x_offset + frame.shape[1]
                    ] = frame
                    cv2.imshow("EyeGaze", image)
                    cv2.waitKey(1)

            else:
                print('not enough data')
                

        def detect_button_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if (
                    x > button_x
                    and x < button_x + button_width
                    and y > button_y
                    and y < button_y + button_height
                ):
                    eyes()

                ## call function if outside of button
                else:
                    mouse_callback(event, x, y, flags, param)

        new_row_list = [int(avg_x * frame_w), int(avg_y * frame_h)]
        cv2.setMouseCallback("EyeGaze", detect_button_click, new_row_list)

        cv2.circle(frame, (int(avg_x * frame_w), int(avg_y * frame_h)), 3, (0, 255, 0))

    x_offset = y_offset = 10
    image[
        y_offset : y_offset + frame.shape[0], x_offset : x_offset + frame.shape[1]
    ] = frame

    cv2.imshow("EyeGaze", image)
    cv2.waitKey(1)
