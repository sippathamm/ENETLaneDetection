from datetime import datetime
import cv2

capture = cv2.VideoCapture(0)  # To use Realsense camera in RGB mode, change index to 4

if not capture.isOpened():
    print('[ERROR] Could not open video device')
    exit()

video_writer1 = cv2.VideoWriter('recorded_raw.mp4', cv2.VideoWriter_fourcc(*"MP4V"), 30, (512, 256))
video_writer2 = cv2.VideoWriter('recorded_timestamp.mp4', cv2.VideoWriter_fourcc(*"MP4V"), 30, (512, 256))

try:
    while True:
        ret, frame = capture.read()
        resized_frame = cv2.resize(frame, (512, 256))
        timestamp_frame = resized_frame.copy()

        if not ret:
            print('[ERROR] Could not read frame')
            break

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        cv2.putText(timestamp_frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video_writer1.write(resized_frame)
        video_writer2.write(timestamp_frame)

        cv2.imshow('Frame', timestamp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print('\n[INFO] KeyboardInterrupt detected.')

finally:
    capture.release()
    video_writer1.release()
    video_writer2.release()
    cv2.destroyAllWindows()

    print('[INFO] Output saved.')
