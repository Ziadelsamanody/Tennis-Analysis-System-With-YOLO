from ultralytics import YOLO
import cv2 as cv 
import pickle 

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def detect_frames(self, frames, read_from_stub=False,  stub_path = None):
        player_detection = []
        if  read_from_stub  and stub_path is not None : 
            with open(stub_path, "rb") as  f :
                player_detection = pickle.load(f)
                
        for frame in frames : 
            player_dict = self.detect_frame(frame)
            player_detection.append(player_dict)
        if stub_path is not None :
            with open(stub_path, "wb") as f :
                pickle.dump(player_detection, f)

        return player_detection
    def detect_frame(self, frame):
        result = self.model.track(frame, persist=True)[0] #presist other frames not just one
        id_names_dict = result.names

        player_dict = {}
    #result.boxes :   
    #cls: tensor([0.]) conf: tensor([0.9049])id: tensor([1.]) ...
#
        for box in result.boxes:  #   
            track_id = int(box.id.item())
            result = box.xyxy[0].tolist()
            object_cls_id= box.cls.tolist()[0]
            object_cls_name = id_names_dict[object_cls_id]

            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict 
    
    def draw_boxes(self, video_frames, player_detection):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detection):
            # Draw bounding boxes
            for track_id , bbox in player_dict.items() : 
                x1,y1,x2,y2 = bbox
                cv.putText(frame, f"Player ID :{track_id}", (int(bbox[0]), int(bbox[1] - 10)),cv.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,0), 2)
                cv.rectangle(frame, (int(x1), int(y1)),(int(x2), int(y2)), (255,0,0), 2)
            output_frames.append(frame)
        return output_frames



#player dict = {track_id : [x1,y1,x2,y2]}
