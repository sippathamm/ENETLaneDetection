from transform import homogeneous_transform, warp_and_combine_image
import numpy as np
import collections
import cv2


class NotFittedError(AttributeError):
    pass


class NotBinaryImageError(AttributeError):
    pass


class NotReceiveImageError(AttributeError):
    pass


class NotCalculateLookaheadError(AttributeError):
    pass


class Lane:
    def __init__(self, reference_point=None, lookahead_gain=0.5, use_mean_fit_coeffs=True, buffer=10):
        self.binary_warped_image = None
        self.image_width = None
        self.image_height = None

        # States
        self.received_image = False
        self.fitted = False
        self.calculated_lookahead = False

        # Camera perspective
        self.reference_pt_c = reference_point
        # Bird's eye perspective
        self.lookahead_pt_be = None
        self.reference_pt_be = None

        self.lookahead_gain = np.clip(lookahead_gain, 0.0, 1.0)
        self.use_mean_fit_coeffs = use_mean_fit_coeffs
        self.left_lane_fit_coeffs = None
        self.right_lane_fit_coeffs = None
        self.recent_left_lane_fit_coeffs = collections.deque([], maxlen=buffer)
        self.recent_right_lane_fit_coeffs = collections.deque([], maxlen=buffer)

        self.left_lane_px_u = None
        self.left_lane_px_v = None
        self.right_lane_px_u = None
        self.right_lane_px_v = None
        self.v = None
        self.lookahead_v = None
        self.ym_per_px = 4 / 720  # meters per pixel in v dimension
        self.xm_per_px = 3.7 / 700  # meters per pixel in x dimension

        self.angle_offset = 0.
        self.center_lane_offset = 0

    def put_warped_image(self, binary_warped_image):
        self.__check_image_is_binary(binary_warped_image)

        self.binary_warped_image = binary_warped_image
        self.image_width = binary_warped_image.shape[1]
        self.image_height = binary_warped_image.shape[0]
        self.v = np.linspace(0, self.image_height - 1, self.image_height)
        self.lookahead_v = int(self.image_height * (1 - self.lookahead_gain))

        self.received_image = True

    def fit_lane(self):
        self.__check_already_received_image()

        self.find_lane_px(margin=30)

        # Fit 2-nd order polynomial
        left_lane_fit_coeffs = np.polyfit(self.left_lane_px_v, self.left_lane_px_u, 2)
        right_lane_fit_coeffs = np.polyfit(self.right_lane_px_v, self.right_lane_px_u, 2)
        self.recent_left_lane_fit_coeffs.append(left_lane_fit_coeffs)
        self.recent_right_lane_fit_coeffs.append(right_lane_fit_coeffs)

        if self.use_mean_fit_coeffs:
            # Use mean fit coefficients
            self.left_lane_fit_coeffs = np.mean(self.recent_left_lane_fit_coeffs, axis=0)
            self.right_lane_fit_coeffs = np.mean(self.recent_right_lane_fit_coeffs, axis=0)
        else:
            # Use latest fit coefficients
            self.left_lane_fit_coeffs = left_lane_fit_coeffs
            self.right_lane_fit_coeffs = right_lane_fit_coeffs

        self.fitted = True

        return self.left_lane_fit_coeffs, self.right_lane_fit_coeffs

    def find_lane_px(self, n_windows=9, margin=100, min_px=50):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped_image[self.image_height // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int32(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = n_windows
        # Set the width of the windows +/- margin
        margin = margin
        # Set minimum number of pixels found to recenter window
        minpix = min_px

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int32(self.image_height // nwindows)
        # Identify the x and v positions of all nonzero pixels in the image
        nonzero = self.binary_warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and v (and right and left)
            win_y_low = self.image_height - (window + 1) * window_height
            win_y_high = self.image_height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and v within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        left_lane_u = nonzerox[left_lane_inds]
        left_lane_v = nonzeroy[left_lane_inds]
        right_lane_u = nonzerox[right_lane_inds]
        right_lane_v = nonzeroy[right_lane_inds]

        self.left_lane_px_u, self.left_lane_px_v = left_lane_u, left_lane_v
        self.right_lane_px_u, self.right_lane_px_v = right_lane_u, right_lane_v

        return self.left_lane_px_u, self.left_lane_px_v, self.right_lane_px_u, self.right_lane_px_v

    def __calculate_lookahead_pt(self):
        self.__check_is_fitted()

        left_lane_u = (self.left_lane_fit_coeffs[0] * self.lookahead_v ** 2 +
                       self.left_lane_fit_coeffs[1] * self.lookahead_v +
                       self.left_lane_fit_coeffs[2])
        right_lane_u = (self.right_lane_fit_coeffs[0] * self.lookahead_v ** 2 +
                        self.right_lane_fit_coeffs[1] * self.lookahead_v +
                        self.right_lane_fit_coeffs[2])

        lookahead_pt = np.array([(left_lane_u + right_lane_u) // 2, self.lookahead_v])
        self.calculated_lookahead = True

        self.lookahead_pt_be = lookahead_pt

    def calculate_angle(self, T):
        self.__calculate_lookahead_pt()
        self.__check_reference_pt_is_none()

        # Transform reference point in car perspective to bird's eye perspective
        self.reference_pt_be = homogeneous_transform(T, self.reference_pt_c)
        # print('lookahead point:', self.lookahead_pt_be)
        # print('reference point:', self.reference_pt_be)

        lookahead_u, lookahead_v = self.lookahead_pt_be[0], self.lookahead_pt_be[1]
        reference_u, reference_v = self.reference_pt_be[0], self.reference_pt_be[1]

        # Angle between reference point and lookahead point
        angle_offset = np.arctan2(reference_u - lookahead_u, reference_v - lookahead_v)

        self.angle_offset = angle_offset

        return self.angle_offset

    def calculate_center_lane_offset(self):
        left_lane_u = (self.left_lane_fit_coeffs[0] * self.image_height ** 2 +
                       self.left_lane_fit_coeffs[1] * self.image_height +
                       self.left_lane_fit_coeffs[2])
        right_lane_u = (self.right_lane_fit_coeffs[0] * self.image_height ** 2 +
                        self.right_lane_fit_coeffs[1] * self.image_height +
                        self.right_lane_fit_coeffs[2])

        center_lane = (left_lane_u + right_lane_u) // 2

        self.center_lane_offset = (self.image_width // 2) - center_lane
        print('center_lane_offset', self.center_lane_offset)

        return self.center_lane_offset

    def get_lane_line_image(self,
                            left_lane_line_color=(0, 255, 0),
                            right_lane_line_color=(0, 255, 0),
                            road_color=(0, 0, 255),
                            line_width=10):
        self.__check_is_fitted()

        left_lane_fit_u = (self.left_lane_fit_coeffs[0] * self.v ** 2 +
                           self.left_lane_fit_coeffs[1] * self.v +
                           self.left_lane_fit_coeffs[2])
        right_lane_fit_u = (self.right_lane_fit_coeffs[0] * self.v ** 2 +
                            self.right_lane_fit_coeffs[1] * self.v +
                            self.right_lane_fit_coeffs[2])

        zero_image = np.zeros_like(self.binary_warped_image).astype(np.uint8)
        lane_line_image = np.dstack((zero_image, zero_image, zero_image))

        # Draw lane lines
        if left_lane_line_color is not None:
            left_lane_line_min = np.array([np.transpose(np.vstack([left_lane_fit_u - line_width // 2, self.v]))])
            left_lane_line_max = np.array(
                [np.flipud(np.transpose(np.vstack([left_lane_fit_u + line_width // 2, self.v])))])
            left_lane_line = np.hstack((left_lane_line_min, left_lane_line_max))
            cv2.fillPoly(lane_line_image, np.int_([left_lane_line]), left_lane_line_color[::-1])
        if right_lane_line_color is not None:
            right_lane_line_min = np.array([np.transpose(np.vstack([right_lane_fit_u - line_width // 2, self.v]))])
            right_lane_line_max = np.array(
                [np.flipud(np.transpose(np.vstack([right_lane_fit_u + line_width // 2, self.v])))])
            right_lane_line = np.hstack((right_lane_line_min, right_lane_line_max))
            cv2.fillPoly(lane_line_image, np.int_([right_lane_line]), right_lane_line_color[::-1])

        # Draw center lane line
        center_lane_fit_u = (left_lane_fit_u + right_lane_fit_u) // 2
        center_lane_line_min = np.array([np.transpose(np.vstack([center_lane_fit_u - line_width // 2, self.v]))])
        center_lane_line_max = np.array(
            [np.flipud(np.transpose(np.vstack([center_lane_fit_u + line_width // 2, self.v])))])
        center_lane_line = np.hstack((center_lane_line_min, center_lane_line_max))
        cv2.fillPoly(lane_line_image, np.int_([center_lane_line]), (255, 255, 255))

        # Draw road
        if road_color is not None:
            left_road_bound = np.array([np.transpose(np.vstack([left_lane_fit_u, self.v]))])
            right_road_bound = np.array([np.flipud(np.transpose(np.vstack([right_lane_fit_u, self.v])))])
            road = np.hstack((left_road_bound, right_road_bound))
            cv2.fillPoly(lane_line_image, np.int_([road]), road_color[::-1])

        return lane_line_image

    def get_lookahead_image(self,
                            triangle_color=(255, 255, 0),
                            reference_pt_color=(255, 0, 0),
                            lookahead_pt_color=(0, 255, 0)):
        self.__check_already_calculate_lookahead()

        zero_image = np.zeros_like(self.binary_warped_image).astype(np.uint8)
        lookahead_image = np.dstack((zero_image, zero_image, zero_image))

        reference_u, reference_v = self.reference_pt_be[0], self.reference_pt_be[1]
        lookahead_u, lookahead_v = self.lookahead_pt_be[0], self.lookahead_pt_be[1]

        pts = np.array([[lookahead_u, lookahead_v],
                        [reference_u, lookahead_v],
                        [reference_u, reference_v]], np.int32).reshape((-1, 1, 2))

        cv2.polylines(lookahead_image, [pts], True, triangle_color[::-1], 2)

        cv2.circle(lookahead_image, (int(reference_u), int(reference_v)),
                   5, reference_pt_color[::-1], -1)

        cv2.circle(lookahead_image, (int(lookahead_u), int(lookahead_v)),
                   5, lookahead_pt_color[::-1], -1)

        return lookahead_image

    @staticmethod
    def project_on_image(original_image,
                         lane_lane_image=None,
                         lookahead_image=None,
                         angle_offset=None,
                         T_inv=None):
        result = original_image

        if lane_lane_image is not None:
            result = warp_and_combine_image(lane_lane_image, original_image,
                                            (int(original_image.shape[1]), int(original_image.shape[0])),
                                            T_inv,
                                            alpha=1, beta=1)
        if lookahead_image is not None:
            result = warp_and_combine_image(lookahead_image, result,
                                            (int(result.shape[1]), int(result.shape[0])),
                                            T_inv, beta=0.8)

        if angle_offset is not None:
            cv2.putText(result, 'angle offset [rad]: ' + str(angle_offset)[:7],
                        (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)

        return result

    @staticmethod
    def __check_image_is_binary(input_image):
        if len(input_image.shape) > 2:
            raise NotBinaryImageError(f'This put image with shape of {input_image.shape}, is not a binary image.')

    def __check_already_received_image(self):
        if not self.received_image:
            raise NotReceiveImageError('This Lane instance does not receive a binary warped image yet. Call '
                                       '\'put_warped_image\' with proper arguments first.')

    def __check_reference_pt_is_none(self):
        if self.reference_pt_c is None:
            self.reference_pt_c = (self.image_width // 2, self.image_height)

    def __check_is_fitted(self):
        if not self.fitted:
            raise NotFittedError('This Lane instance is not fitted yet. Call \'fit_lane\' first.')

    def __check_already_calculate_lookahead(self):
        if not self.calculated_lookahead:
            raise NotCalculateLookaheadError('This Lane instance does not calculate lookahead yet. Call '
                                             '\'calculate_alpha_delta\' with proper arguments first.')
