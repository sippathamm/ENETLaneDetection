from transform import bird_eyes
from lane import Lane
import numpy as np
import cv2

REFERENCE_POS = (512 // 2, 237)
LOOKAHEAD_GAIN = 0.5

# Source points taken from images with straight lane lines, these are to become parallel after the warp transform
# source_points = np.float32([
#     [11, 237],  # bottom-left
#     [191, 119],  # top-left
#     [305, 119],  # top-right
#     [423, 237]  # bottom-right
# ])

source_points = np.float32([
    [-978.8445, 237.],  # bottom-left
    [-16.978928, 128.10956],  # top-left
    [513.02106, 128.10956],  # top-right
    [1474.8866, 237.]  # bottom-right
])

vertices = np.array([
    [170, 0],
    [330, 0],
    [330, 256],
    [170, 256],
], dtype=np.int32)
roi = np.zeros((256, 512), dtype=np.uint8)
cv2.fillPoly(roi, [vertices], (255, 255, 255))

lanes = Lane(reference_point=REFERENCE_POS, lookahead_gain=LOOKAHEAD_GAIN,
             use_mean_fit_coeffs=True, buffer=10)


def postprocessing_pipeline(frame, binary_mask_image):
    global lanes

    binary_warped_image, T, T_inv = bird_eyes(binary_mask_image, source_points, offset=0)
    cv2.imshow('Bird\'s Eye', binary_warped_image)
    # T     is a transformation matrix that transforms an image from camera view to bird eyes view
    # T_inv is an inverse transformation matrix of T

    cut_off = cv2.bitwise_and(binary_warped_image, roi, mask=roi)

    lanes.put_warped_image(cut_off)
    left_lane_fit_coeffs, right_lane_fit_coeffs = lanes.fit_lane()
    lane_line_image = lanes.get_lane_line_image(road_color=None, line_width=7)
    # angle_offset_deg = lanes.calculate_angle(T)
    # lookahead_image = lanes.get_lookahead_image()
    lane_info_image = lanes.project_on_image(original_image=frame,
                                             lane_lane_image=lane_line_image,
                                             T_inv=T_inv)

    return lane_line_image, lane_info_image, \
        left_lane_fit_coeffs, right_lane_fit_coeffs


def main():
    pass


if __name__ == "__main__":
    main()
