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
    
    def get_ball_shots_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1', 'x2', 'y2'])
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions["mid_y_rolling_mean"].diff()
        df_ball_positions['ball_hit'] = 0 
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions ) - int(minimum_change_frames_for_hit *1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0  and df_ball_positions["delta_y"].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[ i + 1] > 0

            if positive_position_change or negative_position_change :
                change_count = 0 
                for change_frame in range(i +1,  i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0  and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame : 
                            change_count += 1
                    elif positive_position_change and positive_position_change_following_frame :
                            change_count += 1
                if change_count > minimum_change_frames_for_hit - 1 : 
                    df_ball_positions['ball_hit'].iloc[i] = 1
        frame_numns_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_numns_with_ball_hits
    
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



    