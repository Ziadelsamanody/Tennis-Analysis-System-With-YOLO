import cv2 as cv
from ultralytics import YOLO
import pickle
import numpy as np
import pandas as pd 




class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, []) for x  in ball_positions]
        # convert  position to df to easy for interpolate
        df_ball_position = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        # interpolate the missing values 
        df_ball_position = df_ball_position.interpolate()
        df_ball_position.bfill(inplace= True)
        #convert this back to same format 
        ball_positions = [{1 : x} for x in df_ball_position.to_numpy().tolist()]
        
        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detection = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f :
                ball_detection = pickle.load(f)
            return ball_detection
        
        for frame in frames :
            result = self.detect_frame(frame)
            ball_detection.append(result)

        if stub_path is not None :
            with open(stub_path, "wb") as f :
                pickle.dump(ball_detection, f)
        return ball_detection

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        # id_names = result.names
        ball_detection = {}

        for box in results.boxes:
            result = box.xyxy[0].tolist()

            ball_detection[1] = result
        return ball_detection
            
    def draw_boxes(self, frames, ball_detection):
        output_video_frames = []
        for frame, ball_detect in zip(frames, ball_detection):
            for id, box in ball_detect.items():
                x1,y1,x2,y2 = box 
                cv.putText(frame, f"Ball ID : {id}", (int(box[0]), int(box[1] - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            output_video_frames.append(frame)
        return output_video_frames
