from datetime import datetime
import serial
import csv

serial = serial.Serial(port='/dev/tty.usbmodem2103', baudrate=115200, timeout=1)

csv_file = open(file='steering_angle_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'steering_angle_rad'])

try:
    while True:
        if serial.in_waiting > 0:
            serial_data = serial.readline().decode('utf-8').strip()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            csv_writer.writerow([timestamp, serial_data])

            print(f'> {timestamp}: {serial_data}')

except KeyboardInterrupt:
    print('\n[INFO] KeyboardInterrupt detected.')

finally:
    serial.close()
    csv_file.close()

    print('[INFO] Output saved.')
