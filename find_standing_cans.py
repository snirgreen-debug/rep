import matplotlib.pyplot as plt
import yaml
import numpy as np
import argparse
import cv2
import realsense_depth


# ============= NOT IN THE FINAL CODE!!!! =================
def mouse_points(event, cursor_x, cursor_y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(cursor_x, cursor_y)


# ============= END OF NOT IN THE FINAL CODE!!!! ==========

class PixCmConverter:
    '''
    Find the pixels to cm ratio.
    assumptions: this ratio is linear in the distance from the "floor".
    '''

    def __init__(self, depth_frame, image_shape):
        self.depth_frame = depth_frame
        self.height, self.width = image_shape

        # find distance from floor
        self.depth_mat = np.array(
            [[self.depth_frame.get_distance(j, i) for j in range(self.width // 4, 3 * self.width // 4)] for i in
             range(self.height // 4, 3 * self.height // 4)]
        )
        self.dist_from_floor = np.max(self.depth_mat)

        # Given the assumptions and empirical measurements, this is the pix:cm ratio
        pix_cm_ratio = -22 * self.dist_from_floor + 23.6
        cm_pix_ratio = 1 / pix_cm_ratio
        self.pix_per_cm = lambda cm: pix_cm_ratio * cm
        self.cm_per_pix = lambda pix: cm_pix_ratio * pix


class TinsIdentifier:
    '''
    this class is responsible for identifying the standing cans using hough transform for circles
    '''
    DP = 1.5
    CANNY1 = 200
    CANNY2 = 50
    MIN_CAN_RADIUS = 2  # in cm
    MAX_CAN_RADIUS = 10  # in cm
    MIN_DIST_BETWEEN_CIRCLES = 7  # in cm

    def __init__(self, color_image, depth_frame, dp=DP, canny1=CANNY1, canny2=CANNY2, min_can_radius=MIN_CAN_RADIUS,
                 max_can_radius=MAX_CAN_RADIUS, min_dist_between_circles=MIN_DIST_BETWEEN_CIRCLES):
        self.color_image = color_image
        self.depth_frame = depth_frame
        self.dp = dp
        self.canny1 = canny1
        self.canny2 = canny2

        # find distances matrix (depth_mat) and find pix:cm ratio
        pix_cm_converter = PixCmConverter(self.depth_frame, self.color_image.shape[:2])
        self.depth_mat = pix_cm_converter.depth_mat
        self.pix_per_cm = pix_cm_converter.pix_per_cm
        self.cm_per_pix = pix_cm_converter.cm_per_pix

        # calculate distances in pixels
        self.min_can_radius = int(self.pix_per_cm(min_can_radius))
        self.max_can_radius = int(self.pix_per_cm(max_can_radius))
        self.min_dist_between_circles = int(self.pix_per_cm(min_dist_between_circles))
        print(self.min_can_radius, self.max_can_radius, self.min_dist_between_circles)
# ===================== NOT in the code =========================
    def show_circles(self, circles):
        img = self.color_image.copy()
        for x, y, r in circles:
            img = cv2.circle(self.color_image, (x, y), r, (0, 255, 0), 1)
        cv2.imshow("img", img)
        cv2.setMouseCallback("img", mouse_points)
        cv2.waitKey(0)
# ===================== END NOT in the code =========================

    def get_bins_points(self, color_code=cv2.COLOR_RGB2GRAY):
        self.grey_image = cv2.cvtColor(self.color_image, color_code)
        big_circles = cv2.HoughCircles(self.grey_image, cv2.HOUGH_GRADIENT, self.dp, self.min_dist_between_circles,
                                       param1=self.canny1, param2=self.canny2,
                                       minRadius=self.min_can_radius, maxRadius=self.max_can_radius).astype(int)[0]
        # ===================== NOT in the code =========================
        self.show_circles(big_circles)
        # ===================== END NOT in the code =========================
        print(big_circles)


if __name__ == "__main__":
    cam = realsense_depth.DepthCamera()
    _, depth_image, color_image, depth_frame, color_frame = cam.get_frame(align_image=True)
    tf = TinsIdentifier(color_image, depth_frame)
    tf.get_bins_points()