from utils import (read_video, save_video) 
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import warnings
warnings.filterwarnings("ignore")
def main():
    # read video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    #detect players 
    player_tracker = PlayerTracker('yolov8x.pt')
    player_detection = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detection.pkl')
  
    
    # detect ball 
    ball_tracker = BallTracker('models/yolov5last.pt')
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/ball_detection.pkl')


    # Court line detector
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0]) # only video frames

    #draw output 
    #draw player and ball  Bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_boxes(video_frames, ball_detection)

    # Draw court keypoints 
    output_video_frames =  court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()