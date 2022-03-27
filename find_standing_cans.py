import numpy as np
import itertools
import cv2
import realsense_depth
import random


# ============= NOT IN THE FINAL CODE!!!! =================
def mouse_points(event, cursor_x, cursor_y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(cursor_x, cursor_y)


# ============= END OF NOT IN THE FINAL CODE!!!! ==========

class PixCmConverter:
    """
    Find the pixels to cm ratio.
    assumptions: this ratio is linear in the distance from the "floor".
    """

    def __init__(self, pix_depth_frame, image_shape, depth_matrix):
        self.depth_frame = pix_depth_frame
        self.height, self.width = image_shape

        # find distance from floor
        self.depth_mat = depth_matrix[self.width // 4:3 * self.width // 4, self.height // 4:3 * self.height // 4]
        self.depth_mat[self.depth_mat > np.percentile(self.depth_mat, 90)] = np.median(
            self.depth_mat)  # capping outliers
        self.dist_from_floor = np.max(self.depth_mat)

        # Given the assumptions and empirical measurements, this is the pix:cm ratio
        pix_cm_ratio = -22 * self.dist_from_floor + 23.6
        cm_pix_ratio = 1 / pix_cm_ratio
        self.cm_to_pix = lambda cm: pix_cm_ratio * cm
        self.pix_to_cm = lambda pix: cm_pix_ratio * pix


class BitsIdentifier:
    """
    BitsIdentifier identifies the bits given a tin can's mask(circle)
    """
    DP = 1.5
    # O_CANNYS = [[100, 22], [100, 24], [100, 26], [100, 28], [100, 30], [100, 32], [100, 34], [100, 36], [100, 42],
    #             [110, 22], [110, 24], [110, 26], [110, 28], [110, 30], [110, 32], [110, 34], [110, 36], [110, 40],
    #             [120, 20], [120, 22], [120, 24], [120, 26], [120, 28], [120, 30], [120, 32], [120, 34], [120, 36],
    #             [130, 14], [130, 18], [130, 20], [130, 22], [130, 24], [130, 26], [130, 28], [130, 30], [140, 16],
    #             [140, 18], [140, 20], [140, 22], [140, 24], [140, 26], [140, 28], [140, 30], [150, 18], [150, 20],
    #             [150, 22], [150, 24], [150, 26], [150, 28], [150, 30], [160, 16], [80, 26], [80, 28], [80, 30],
    #             [80, 34],
    #             [80, 36], [80, 38], [80, 40], [90, 22], [90, 24], [90, 26], [90, 28], [90, 30], [90, 32], [90, 34],
    #             [90, 36]]
    O_CANNYS = tuple([[100, 26], [100, 28], [100, 30], [100, 32], [100, 34], [100, 36], [110, 26],
                      [110, 28], [110, 30], [110, 32], [110, 34], [110, 36], [120, 26], [120, 28], [120, 30],
                      [120, 32], [120, 34], [120, 36], [130, 26], [130, 28], [130, 30], [140, 26],
                      [140, 28], [140, 30], [150, 24], [150, 26], [150, 28], [150, 30], [80, 34], [80, 36], [80, 38],
                      [80, 40], [90, 26], [90, 28], [90, 30], [90, 32], [90, 34], [90, 36]])
    CANNYS = O_CANNYS
    MIN_CAN_RADIUS = 1  # in px
    MAX_CAN_RADIUS = 1  # in cm
    MIN_DIST_BETWEEN_CIRCLES = 4  # in cm
    PIXELS_FOR_MEAN = 4

    def __init__(self, bit_color_image, can, bit_depth_frame, color_code=cv2.COLOR_RGB2GRAY, dp=DP, cannys=CANNYS,
                 min_can_radius_in_pix=MIN_CAN_RADIUS, max_can_radius_in_cm=MAX_CAN_RADIUS,
                 min_dist_between_circles_in_cm=MIN_DIST_BETWEEN_CIRCLES, pixels_for_mean=PIXELS_FOR_MEAN):
        """
        Initialize - define a converter, the needed images (color, gray, depth matrix), and radius calculation
        """
        self.color_image = bit_color_image
        self.gray_image = cv2.cvtColor(self.color_image, color_code)
        self.depth_frame = bit_depth_frame

        height, width = self.color_image.shape[:2]
        self.depth_mat = self.depth_mat = np.array(
            [[self.depth_frame.get_distance(j, i) for j in range(width)] for i in
             range(height)]
        )

        self.pixels_for_mean = pixels_for_mean
        self.can = can
        self.dp = dp
        self.cannys = cannys

        # find distances matrix (depth_mat) and find pix:cm ratio
        self.pix_cm_converter = PixCmConverter(self.depth_frame, self.color_image.shape[:2], self.depth_mat)
        self.pix_per_cm = self.pix_cm_converter.cm_to_pix
        self.cm_per_pix = self.pix_cm_converter.pix_to_cm

        self.min_can_radius = min_can_radius_in_pix
        self.max_can_radius = int(self.pix_cm_converter.cm_to_pix(max_can_radius_in_cm))
        self.min_dist_between_circles = int(self.pix_cm_converter.cm_to_pix(min_dist_between_circles_in_cm))

    def set_mask(self):
        """
        Given the can's circle, return the masked image (gray image containing only the circle)
        """
        gray_image = self.gray_image.copy()
        mask = np.zeros(gray_image.shape, np.uint8)
        mask = cv2.circle(mask, self.can[:2], self.can[2], 1, -1)
        return mask * gray_image

    def get_mean_distance(self, point):
        """
        Given a point, calculate the mean distance from the point.
        This method is for more accurate result for surface.
        """
        x, y = point
        pix_for_mean = self.pixels_for_mean
        height, width = self.depth_mat.shape
        surface = self.depth_mat.copy()
        surface = surface[
                  max(y - pix_for_mean, 0):min(y + pix_for_mean, height),
                  max(x - pix_for_mean, 0):min(x + pix_for_mean, width)
                  ]
        surface = np.nan_to_num(surface)
        surface[(surface > np.percentile(surface, 75)) | (surface < np.percentile(surface, 25))] = np.median(
            surface)  # capping outliers
        result = np.median(surface)
        return result

    def __circle_heuristic(self, circle, count, distance_from_center, radius, avg_point, min_count, max_count,
                           close_count, close_count_max):
        """
        Heuristics for choosing the bit's circle, if there are few results.
        The grade is given by:
            1. Count of where circle was found (in the canny images).
            2. distance from the midpoint.
            3. how close is the circle from the other circle (usually, the bit is close to other circles).
            4. distance from the center (should be about 3/4 * radius).
        """
        norm_count = (count - min_count) / (max_count - min_count)
        norm_close_count = close_count / close_count_max
        dist_from_center_factor = abs(distance_from_center - 3 * radius / 4) / (3 * radius / 4)
        dist_from_mean = np.linalg.norm(circle - avg_point)
        dist_from_mean = 1 if dist_from_mean == 0 else 1 / dist_from_mean
        is_legal = 0.3 * radius < distance_from_center < 0.95 * radius
        grade = 0.5 * norm_count + 0.3 * dist_from_mean + 0.13 * dist_from_center_factor + 0.07 * norm_close_count
        grade = is_legal * grade
        return grade

    def get_bit_circle(self):
        """
        Find the bit circles in the masked image
        """
        CLOSE_DIST = 20

        masked_image = self.set_mask().copy()
        small_circles = None

        # Run HoughCircle with different parameters and get the most frequent circles, which is probably the bit.
        for canny1, canny2 in self.cannys:
            current_small_circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, self.dp,
                                                     self.min_dist_between_circles,
                                                     param1=canny1, param2=canny2,
                                                     minRadius=self.min_can_radius, maxRadius=self.max_can_radius)

            if current_small_circles is None or len(current_small_circles) == 0:
                continue
            current_small_circles = current_small_circles.astype(int)[0]
            small_circles = current_small_circles if small_circles is None else np.concatenate((
                small_circles, current_small_circles
            ))

        if small_circles is None:
            return None

        # Get the most likely small circle by frequency
        unique_circles, unique_count = np.unique(small_circles, axis=0, return_counts=True)
        for i in unique_circles:
            x, y, r = i
            self.color_image = cv2.circle(self.color_image, (x, y), r, (255, 255, 0), 1)
        cv2.imshow("a", self.color_image)
        cv2.setMouseCallback("a", mouse_points)
        cv2.waitKey(0)
        real_count = np.zeros(unique_count.shape)
        for i, unique_circle in enumerate(unique_circles):
            for j, other_circle in enumerate(unique_circles):
                dist = np.linalg.norm(unique_circle[:2] - other_circle[:2])
                if dist < CLOSE_DIST:
                    dist = 1 if dist == 0 else dist
                    real_count[i] += unique_count[j] / dist
        unique_count = real_count

        mean_point = np.mean(unique_circles[:, :2], axis=0)
        heuristics = np.zeros(unique_count.shape)
        min_count = np.min(unique_count)
        max_count = np.max(unique_count)
        for i in range(len(unique_circles)):
            circle = unique_circles[i][:2]
            distance = np.linalg.norm(self.can[:2] - circle)
            circles_distances = np.apply_along_axis(lambda i: np.linalg.norm(i - circle), 1, unique_circles[:, :2])
            close_count = np.count_nonzero(circles_distances < CLOSE_DIST)
            count = unique_count[i]
            heuristics[i] = self.__circle_heuristic(circle, count, distance, self.can[2], mean_point, min_count,
                                                    max_count, close_count, len(unique_circles))
        if np.max(heuristics) == 0:
            return None
        return unique_circles[np.argmax(heuristics)]

    def __find_mc(self, points):
        """
        Find m, c in the line equation given two points.
        """
        x_coords, y_coords = zip(*points)
        a = np.vstack([x_coords, np.ones(len(x_coords))]).T
        return np.linalg.lstsq(a, y_coords)[0]

    def __get_quad_formula_result(self, a, b, c):
        """
        This is the solution of quadratic equation.
        """
        delta = np.sqrt(np.power(b, 2) - 4 * a * c)
        denominator = 2 * a
        return (-b + delta) / denominator, (-b - delta) / denominator

    def __get_intersection_points(self, a, b, r, m, c):
        """
        Get the intersection points between bit and line
        """
        xs = self.__get_quad_formula_result(
            1 + np.power(m, 2),
            2 * (m * c - m * b - a),
            np.power(a, 2) + np.power(b, 2) + np.power(c, 2) - np.power(r, 2) - 2 * b * c
        )
        ys = tuple(map(lambda x: m * x + c, xs))
        return tuple(zip(xs, ys))

    def get_sweet_spot(self, bit_circle):
        """
        Find the best point to lift the can
        """
        if bit_circle is None:
            return self.can[:2]
        can_a, can_b, can_r = self.can
        bit_a, bit_b, bit_r = bit_circle

        p1 = (can_a, can_b)
        p2 = (bit_a, bit_b)
        points = [p1, p2]
        m, c = self.__find_mc(points)

        # Intersection points:
        bit_ips = np.array(self.__get_intersection_points(bit_a, bit_b, bit_r, m, c))
        bit_ip1, bit_ip2 = bit_ips
        can_ips = np.array(self.__get_intersection_points(can_a, can_b, can_r, m, c))
        distances = np.apply_along_axis(lambda i: np.linalg.norm(i - bit_ip1), 1, can_ips)
        can_point = can_ips[np.argmax(distances)]

        return ((can_point + self.can[:2]) / 2).astype(int)


if __name__ == "__main__":
    # ===================== NOT in the code =========================
    # color_image = cv2.imread("colored_image.jpg")
    # masked_image = cv2.imread("masked_image.jpg")
    # masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("a", masked_image)
    # cv2.waitKey(0)
    # for i in range(80, 500, 10):
    #     for j in range(10, 60, 2):
    #         canny = cv2.Canny(masked_image, i, j)
    #         bla = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, 1.5, 10, None, i, j, 1, 20)
    #         if bla is not None:
    #             bla = bla.astype(int)[0]
    #         it = bla if bla is not None else []
    #         canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    #         for c in it:
    #             canny = cv2.circle(canny, c[:2], c[2], (0, 255, 0), 2)
    #
    #         dir = "./canny_tests/" + ("_".join(['canny1', str(i), 'canny2', str(j)])) + ".jpg"
    #         cv2.imwrite(dir, canny)
    # ===================== END NOT in the code =====================

    cam = realsense_depth.DepthCamera()
    _, depth_image, color_image, depth_frame, color_frame = cam.get_frame(align_image=True)
    bi = BitsIdentifier(color_image, (200, 200, 200), depth_frame)
    bit = bi.get_bit_circle()
    bi.get_mean_distance((200, 200))
    # # ===================== NOT in the code =========================
    # color_image = cv2.circle(color_image, bit[:2], bit[2], (0, 255, 255), 3)
    # cv2.imshow("ci", color_image)
    # cv2.waitKey(0)
    # # ===================== END NOT in the code =====================

    sweet_spot = bi.get_sweet_spot(bit)
    # # ===================== NOT in the code =========================
    print(sweet_spot)
    # print(bit)
    img = color_image.copy()
    img = cv2.circle(img, bit[:2], bit[2], (0, 255, 0), 2)
    img = cv2.circle(img, tuple(sweet_spot), 3, (0, 255, 255), -1)
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", mouse_points)
    cv2.waitKey(0)
    #
    # # ===================== END NOT in the code =====================
