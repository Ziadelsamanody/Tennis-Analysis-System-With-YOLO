import  torch
import  torchvision
from  torchvision import models
from torchvision import transforms
import cv2 as cv
import numpy as np
import torch.nn as  nn 



class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2) #14keypoints with two x, y
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))

        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    def predict(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        # keypoints = outputs.squeeze().tolist().cpu().numpy()
        keypoints = outputs.squeeze().detach().cpu().numpy()


        # return keypoints pos in reall size
        original_h, orginal_w = image.shape[:2]

        keypoints[::2] *= orginal_w / 224.0
        keypoints[1::2] *= original_h  / 224.0

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv.putText(image, str(i // 2), (x, y -10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
            cv.circle(image, (x,y), 5, (255,0,0), -1) #-1 is thicknees be fill
        return image
    

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame  in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

