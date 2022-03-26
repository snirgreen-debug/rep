import numpy as np
import itertools
import cv2
import realsense_depth


# ============= NOT IN THE FINAL CODE!!!! =================
def mouse_points(event, cursor_x, cursor_y):
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
        self.dist_from_floor = np.max(self.depth_mat)

        # Given the assumptions and empirical measurements, this is the pix:cm ratio
        pix_cm_ratio = -22 * self.dist_from_floor + 23.6
        cm_pix_ratio = 1 / pix_cm_ratio
        self.pix_per_cm = lambda cm: pix_cm_ratio * cm
        self.cm_per_pix = lambda pix: cm_pix_ratio * pix


class TinsIdentifier:
    """
    this class is responsible for identifying the standing cans using hough transform for circles
    """
    DP = 1.5
    CANNY1 = 250
    CANNY2 = 45
    MIN_CAN_RADIUS = 2  # in cm
    MAX_CAN_RADIUS = 10  # in cm
    MIN_DIST_BETWEEN_CIRCLES = 7  # in cm
    PIXELS_FOR_MEAN = 10

    def __init__(self, tin_color_image, tin_depth_frame, color_code=cv2.COLOR_RGB2GRAY, dp=DP, canny1=CANNY1,
                 canny2=CANNY2, min_can_radius=MIN_CAN_RADIUS, max_can_radius=MAX_CAN_RADIUS,
                 min_dist_between_circles=MIN_DIST_BETWEEN_CIRCLES, pixels_for_mean=PIXELS_FOR_MEAN):
        self.color_image = tin_color_image
        self.gray_image = cv2.cvtColor(self.color_image, color_code)
        self.depth_frame = tin_depth_frame

        height, width = self.color_image.shape[:2]
        self.depth_mat = self.depth_mat = np.array(
            [[self.depth_frame.get_distance(j, i) for j in range(width)] for i in
             range(height)]
        )

        self.pixels_for_mean = pixels_for_mean
        self.big_circles = None
        self.dp = dp
        self.canny1 = canny1
        self.canny2 = canny2

        # find distances matrix (depth_mat) and find pix:cm ratio
        self.pix_cm_converter = PixCmConverter(self.depth_frame, self.color_image.shape[:2], self.depth_mat)
        self.pix_per_cm = self.pix_cm_converter.pix_per_cm
        self.cm_per_pix = self.pix_cm_converter.cm_per_pix

        # calculate distances in pixels
        self.min_can_radius = int(self.pix_per_cm(min_can_radius))
        self.max_can_radius = int(self.pix_per_cm(max_can_radius))
        self.min_dist_between_circles = int(self.pix_per_cm(min_dist_between_circles))

    # ===================== NOT in the code =========================
    def show_circles(self, circles):
        img2 = self.color_image.copy()
        for x, y, r in circles:
            img2 = cv2.circle(self.color_image, (x, y), r, (0, 255, 0), 1)
        cv2.imshow("img", img2)
        cv2.imshow("canny", cv2.Canny(self.gray_image, self.canny1, self.canny2))
        cv2.setMouseCallback("img", mouse_points)
        cv2.waitKey(0)

    # ===================== END NOT in the code =========================

    def get_cans_circles(self):
        self.big_circles = cv2.HoughCircles(self.gray_image, cv2.HOUGH_GRADIENT, self.dp, self.min_dist_between_circles,
                                            param1=self.canny1, param2=self.canny2,
                                            minRadius=self.min_can_radius, maxRadius=self.max_can_radius).astype(int)[0]
        # ===================== NOT in the code =========================
        self.show_circles(self.big_circles)
        # ===================== END NOT in the code =========================

    def __get_mean_distance(self, c):
        x, y = c
        pix_for_mean = self.pixels_for_mean
        height, width = self.depth_mat.shape
        surface = self.depth_mat.copy()
        surface = surface[
                  max(y - pix_for_mean, 0):min(y + pix_for_mean, height),
                  max(x - pix_for_mean, 0):min(x + pix_for_mean, width)
                  ]
        surface = np.nan_to_num(surface)
        # surface[(surface > np.percentile(surface, 75)) | (surface < np.percentile(surface, 25))] = np.median(surface)
        # result = np.mean(surface)
        result = np.median(surface)
        return result

    def find_upper_can(self):
        if self.big_circles is None:
            return None
        cans_distances = np.apply_along_axis(lambda c: self.__get_mean_distance(c), 1, self.big_circles[:, :2])
        return self.big_circles[np.argmin(cans_distances)]


class BitsIdentifier:
    DP = (1.4, 1.5, 1.6)
    CANNY1 = range(90, 110, 2)
    CANNY2 = range(15, 25, 1)
    MIN_CAN_RADIUS = 1  # in px
    MAX_CAN_RADIUS = 1  # in cm
    MIN_DIST_BETWEEN_CIRCLES = 4  # in cm

    def __init__(self, bit_color_image, can, pix_cm_converter, color_code=cv2.COLOR_RGB2GRAY, dps=DP, canny1s=CANNY1,
                 canny2s=CANNY2, min_can_radius_in_pix=MIN_CAN_RADIUS, max_can_radius_in_cm=MAX_CAN_RADIUS,
                 min_dist_between_circles_in_cm=MIN_DIST_BETWEEN_CIRCLES):
        self.color_image = bit_color_image
        self.gray_image = cv2.cvtColor(self.color_image, color_code)
        self.can = can
        self.dps = dps
        self.canny1s = canny1s
        self.canny2s = canny2s
        self.pix_cm_converter = pix_cm_converter
        self.min_can_radius = min_can_radius_in_pix
        self.max_can_radius = int(self.pix_cm_converter.pix_per_cm(max_can_radius_in_cm))
        self.min_dist_between_circles = int(self.pix_cm_converter.pix_per_cm(min_dist_between_circles_in_cm))

    def set_mask(self):
        gray_image = self.gray_image.copy()
        mask = np.zeros(gray_image.shape, np.uint8)
        mask = cv2.circle(mask, self.can[:2], self.can[2], 1, -1)
        return mask * gray_image

    # ===================== NOT in the code =========================
    def show_circles(self, circles, mask=None):
        img1 = mask if mask is not None else self.color_image.copy()
        for x, y, r in circles:
            img1 = cv2.circle(self.color_image, (x, y), r, (0, 255, 0), 1)
        cv2.imshow("img", img1)
        c1, c2 = np.random.choice(self.canny1s), np.random.choice(self.canny2s)
        cv2.imshow("canny", cv2.Canny(img1, c1, c2))
        cv2.setMouseCallback("img", mouse_points)
        cv2.setMouseCallback("canny", mouse_points)
        cv2.waitKey(0)

    # ===================== END NOT in the code =====================

    def get_bit_circle(self):
        masked_image = self.set_mask().copy()
        small_circles = None

        # Run HoughCircle with different parameters and get the most frequent circles, which is probably the bit.
        for dp, canny1, canny2 in itertools.product(self.dps, self.canny1s, self.canny2s):
            current_small_circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, dp,
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

        # Get the most frequent small circles
        unique_circles, unique_count = np.unique(small_circles, axis=0, return_counts=True)
        unique_circles = unique_circles[unique_count == np.max(unique_count)]

        # Find most distant circle and return it
        circle_origin = np.array(self.can[:2])
        small_origins = unique_circles[:, :2]
        distances = np.apply_along_axis(lambda i: np.linalg.norm(i - circle_origin), 1, small_origins)
        can_bit = unique_circles[np.argmax(distances)]
        return can_bit

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
            np.power(a, 2) + np.power(b, 2) + np.power(c, 2) - np.power(r, 2) - b * c
        )
        ys = tuple(map(lambda x: m * x + c, xs))
        return tuple(zip(xs, ys))

    def get_sweet_spot(self, bit_circle):
        """
        Find the best point to lift the can
        """
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
        distances = np.apply_along_axis(lambda i: np.linalg.norm(i - can_point), 1, bit_ips)
        bit_point = bit_ips[np.argmax(distances)]

        return (can_point + bit_point) / 2


if __name__ == "__main__":
    # ===================== NOT in the code =========================
    # color_image = cv2.imread("colored_image.jpg")
    # masked_image = cv2.imread("masked_image.jpg")
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # for i in range(100, 500, 10):
    #     for j in range(10, 60, 5):
    #         canny = cv2.Canny(masked_image, i, j)
    #         dir = "./canny_tests/" + ("_".join(['canny1', str(i), 'canny2', str(j)])) + ".jpg"
    #         cv2.imwrite(dir, canny)
    # ===================== END NOT in the code =====================

    cam = realsense_depth.DepthCamera()
    _, depth_image, color_image, depth_frame, color_frame = cam.get_frame(align_image=True)
    ti = TinsIdentifier(color_image, depth_frame)
    ti.get_cans_circles()
    upper_can = ti.find_upper_can()
    bi = BitsIdentifier(color_image, upper_can, ti.pix_cm_converter)
    bit = bi.get_bit_circle()
    sweet_spot = bi.get_sweet_spot(bit)
    print(sweet_spot)
    # ===================== NOT in the code =========================
    print(bit)
    img = color_image.copy()
    img = cv2.circle(img, bit[:2], bit[2], (0, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)

    # ===================== END NOT in the code =====================
