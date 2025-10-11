from  utils import (read_video, save_video)
from tracker import PlayerTracker

path = "input/input_video.mp4"


frames = read_video(path=path)

tracker = PlayerTracker(model="yolov8x.pt")

output = tracker.detect_frames(frames)
outputs = tracker.draw_boxes(frames, output)

save_video(outputs, output_path="output_video.avi")