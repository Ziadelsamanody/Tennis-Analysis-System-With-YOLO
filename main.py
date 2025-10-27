from utils import (read_video, save_video) 
from trackers import PlayerTracker, BallTracker

def main():
    # read video
    input_video_path = 'input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    #detect players 
    player_tracker = PlayerTracker('yolov8x.pt')
    player_detection = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='tracker_stubs/player_detection.pkl')
  
    
    # detect ball 
    ball_tracker = BallTracker('models/yolov5last.pt')
    ball_detection = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path='tracker_stubs/ball_detection.pkl')

    #draw output 
    #draw player and ball  Bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_boxes(video_frames, ball_detection)

    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()