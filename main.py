from utils import (read_video, save_video) 
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2 as cv 


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
    ball_detection = ball_tracker.interpolate_ball_position(ball_detection)

    # Court line detector
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0]) # only video frames

    #choeose_players close to court
    player_detection = player_tracker.choose_and_filter_player(court_keypoints, player_detection)

    # mini court
    mini_court = MiniCourt(video_frames[0])

    # detect ball shots 
    ball_shot_frames = ball_tracker.get_ball_shots_frames(ball_detection)
    print(ball_shot_frames)

    # convert the position to mini court position
    player_mini_court_detection , ball_mini_court_detection = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detection,
                                                                                                               ball_detection, court_keypoints)
    #draw output 
    #draw player and ball  Bounding boxes
    output_video_frames = player_tracker.draw_boxes(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_boxes(video_frames, ball_detection)

    # Draw court keypoints 
    output_video_frames =  court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detection)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detection, color=(0,255,255))

    # Draw Frames on top left corner
    for i , frame in enumerate(output_video_frames):
        cv.putText(frame, f"Frame {i}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__ == '__main__':
    main()