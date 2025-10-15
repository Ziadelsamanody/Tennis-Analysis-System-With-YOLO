from ultralytics import YOLO

model = YOLO('models/yolov5last.pt')

model.predict('input_videos/input_video.mp4', save=True)
