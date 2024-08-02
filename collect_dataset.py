import cv2
import serial
import csv
from datetime import datetime

cap = cv2.VideoCapture(0)
video_writer = cv2.VideoWriter('recorded.mp4', cv2.VideoWriter_fourcc(*"MP4V"), 30, (512, 256))

if not cap.isOpened():
    print('[ERROR] Could not open video device')
    exit()

serial_port = serial.Serial('/dev/tty.usbmodem2103', 115200, timeout=1)

csv_file = open('steering_angle_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'steering_angle_rad'])

try:
    while True:
        ret, frame = cap.read()
        resized_frame = cv2.resize(frame, (512, 256))

        if not ret:
            print('[ERROR] Could not read frame')
            break

        if serial_port.in_waiting > 0:
            serial_data = serial_port.readline().decode('utf-8').strip()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Timestamp with milliseconds
            csv_writer.writerow([timestamp, serial_data])
            print(f'> {timestamp}: {serial_data}')

            cv2.putText(resized_frame, timestamp, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            video_writer.write(resized_frame)

        cv2.imshow('Frame', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print('\n[INFO] KeyboardInterrupt detected.')

finally:
    cap.release()
    video_writer.release()
    serial_port.close()
    csv_file.close()
    cv2.destroyAllWindows()

    print('[INFO] Output saved.')