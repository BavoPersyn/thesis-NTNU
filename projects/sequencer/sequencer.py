import math
import cv2
import collections
import os
import video_to_images
import os.path
from os import path
import numpy as np
import random as rand
from math import tan
import matplotlib.pyplot as plt
from copy import copy


def rotate_image(image, angle):
    """
    Rotate image over angle (purely for visualisation purposes)
    :param image: image to rotate
    :param angle: angle over which to rotate
    :return: rotated image
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    R = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, R, (width, height))
    return rotated


def rotation_matrix_to_euler_angles(rotation_matrix):
    """
    Convert rotation matrix to corresponding Euler angles
    :param rotation_matrix: rotation matrix
    :return: Euler angles
    """
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
    x, y, z = math.degrees(x), math.degrees(y), math.degrees(z)

    return np.array([x, y, z])


def reduce_contrast(image):
    """
    Reduce contrast in image to better view keypoints and matches drawn upon image
    :param image: image to reduce contrast of
    :return: contrast reduced image
    """
    mapper = np.vectorize(lambda x: (x * 127) // 255 + 128)
    image = mapper(image).astype(np.uint8)
    return image


def form_transformation_matrix(r, t):
    """
    Form transformation matrix (4x4) from rotation and translation like this:
        r11, r12, r13, t1
        r21, r22, r23, t2
        r31, r32, r33, t3
         0 ,  0 ,  0 , 1
    :param r: rotation matrix (3x3)
    :param t: translation vector (3x1)
    :return: transformation matrix
    """
    T = np.zeros((4, 4))
    T[3][3] = 1
    for i in range(3):
        for j in range(3):
            T[i][j] = r[i][j]
    for i in range(3):
        T[i][3] = t[i]
    return T


def make_rotation_matrix(theta, psi, phi, radians=True):
    """
    Form rotation matrix from Euler angles
    :param theta: Rotation angle around x-axis
    :param psi: Rotation angle around y-axis
    :param phi: Rotation angle around z-axis
    :param radians: Boolean to denote if angles are given in radians or not
    :return: rotation matrix (3x3)
    """
    if not radians:
        theta = math.radians(theta)
        psi = math.radians(psi)
        phi = math.radians(phi)
    rx = np.array(
        [[1, 0, 0],
         [0, math.cos(theta), -math.sin(theta)],
         [0, math.sin(theta), math.cos(theta)]])

    ry = np.array(
        [[math.cos(psi), 0, math.sin(psi)],
         [0, 1, 0],
         [-math.sin(psi), 0, math.cos(psi)]])

    rz = np.array(
        [[math.cos(phi), -math.sin(phi), 0],
         [math.sin(phi), math.cos(phi), 0],
         [0, 0, 1]])
    rzy = np.matmul(rz, ry)
    r = np.matmul(rzy, rx)
    return r


def rot_mat(theta, psi, phi, radians=True):
    """
    Form rotation matrix from Euler angles
    :param theta: Rotation angle around x-axis
    :param psi: Rotation angle around y-axis
    :param phi: Rotation angle around z-axis
    :param radians: Boolean to denote if angles are given in radians or not
    :return: rotation matrix (3x3)
    """
    if not radians:
        theta = math.radians(theta)
        psi = math.radians(psi)
        phi = math.radians(phi)
    c1 = np.cos(theta)
    s1 = np.sin(theta)
    c2 = np.cos(psi)
    s2 = np.sin(psi)
    c3 = np.cos(phi)
    s3 = np.sin(phi)
    matrix = np.array([[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                       [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                       [-s2, c2 * s3, c2 * c3]])
    return matrix


def check_possibility(rotation, translation, normal, points):
    """
    Check if combination of motion parameters is a possible decomposition of homography
    If it is a possible set of motion parameters, check the amount of points that are laying in front of the image plane
    :param rotation: Rotation matrix (3x3)
    :param translation: Translation vector (3x1)
    :param normal: Normal vector (3x1)
    :param points: keypoints (after motion)
    :return: Boolean that states if solution is possible and amount of keypoints that are correct
    """
    # If translation goes negative along z-axis: car is going backwards so not possible
    if translation[2] < 0:
        return False, 0
    n = np.transpose(normal)
    r = np.transpose(rotation)
    nr = np.matmul(n, r)
    nrt = np.matmul(nr, translation)
    test_value = 1 + nrt
    # 1 + n^TR^Tt should be bigger then 0
    if test_value < 0:
        return False, 0
    good_points = 0
    for point in points:
        m = [point[0], point[1], 1]
        test_value = np.matmul(np.transpose(m), normal)
        if test_value > 0:
            good_points += 1
    return True, good_points


def invert_transform_matrix(t_mat):
    """
    Invert transformation matrix:
        Invert rotation by transposing the rotation matrix
        Invert translation by negating the translation vector
    :param t_mat: transformation matrix
    :return: inverted transformation matrix
    """
    rotation = np.zeros((4, 4))
    translation = np.zeros((4, 4))
    # Transpose rotation part of T
    rotation[3][3] = 1
    for i in range(3):
        for j in range(3):
            rotation[i][j] = t_mat[j][i]
    # Make translation part negative
    for i in range(4):
        translation[i][3] = -t_mat[i][3]
        translation[i][i] = 1
    new_t = np.matmul(rotation, translation)
    return new_t


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_horizon_point(p1, p2, x):
    """
    Calculate y-value of point on horizon line (on image) going through p1 and p2
    :param p1: point1
    :param p2: point2
    :param x: x-value of point
    :return: Point with x = x-value on horizon line
    """
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
    y = -(a * x + c)/b
    return np.array([int(x), int(y)])


def point_in_distance(normal, x, z, d):
    """
    Find y-value of point in the distance on plane with given normal vector (given its x and z coordinate and the
    distance of the camera center to the plane)
    :param normal: normal vector of plain
    :param x: x-value of point
    :param z: y-value of point
    :param d: distance of camera to ground plane
    :return: Point in the distance (in homogeneous coordinates)
    """
    a, b, c = normal
    if b == 0:
        return None
    y = -(a * x + c * z + d)/b
    return np.array([x, y[0], z, 1])


def points_on_line(a, b, c, width):
    """
    Finds two points on the line defined by ax + by + c = 0.
    Points are the leftmost and rightmost points on an image with certain width
    :param a:
    :param b:
    :param c:
    :param width: width of image
    :return: array of two points on the line
    """
    x1 = 0
    x2 = width
    y1 = -(a * x1 + c)/b
    y2 = -(a * x2 + c)/b
    return np.array([(int(x1), int(y1)), (int(x2), int(y2))])


def point_in_range(point, a, b, c, threshold=10):
    """
    Checks whether point lays close enough to line defined by ax + bx + c = 0
    :param point: point to check
    :param a:
    :param b:
    :param c:
    :param threshold: max distance to line
    :return: Boolean that states if points lays close enough or not
    """
    x = point[0]
    y = point[1]
    distance = abs(a*x + b*y + c)/math.sqrt(a**2 + b**2)
    return distance <= threshold


def calculate_rotation_angle(rotation):
    """
    Calculate the rotation angle of a rotation matrix
    :param rotation: rotation matrix (3x3)
    :return: rotation angle
    """
    trace = 0
    for i in range(3):
        trace += rotation[i][i]
    if trace > 3:
        trace = math.floor(trace)
    elif trace < -1:
        trace = math.ceil(trace)
    return math.acos((trace-1)/2)


def calculate_rotation_axis(rotation):
    """
    Calculates the rotation axis of a rotation
    :param rotation: rotation matrix (3x3)
    :return: rotation axis
    """
    axis = np.zeros((3, 1))
    axis[0] = rotation[2][1] - rotation[1][2]
    axis[1] = rotation[0][2] - rotation[2][0]
    axis[2] = rotation[1][0] - rotation[0][1]
    return axis/np.linalg.norm(axis)


def rotation_matrix_from_axis_and_angle(axis, angle):
    """
    Calculates the rotation matrix based on the rotation angle and axis
    :param axis: rotation axis (3x1)
    :param angle: angle (in radians)
    :return: rotation matrix (3x3)
    """
    R = np.zeros((3, 3))
    cos = math.cos(angle)
    sin = math.sin(angle)
    ux = axis[0][0]
    uy = axis[1][0]
    uz = axis[2][0]

    R[0][0] = cos + ux**2 * (1 - cos)
    R[0][1] = ux * uy * (1 - cos) - uz * sin
    R[0][2] = ux * uz * (1 - cos) + uy * sin
    R[1][0] = uy * ux * (1 - cos) + uz * sin
    R[1][1] = cos + uy**2 * (1 - cos)
    R[1][2] = uy * uz * (1 - cos) - ux * sin
    R[2][0] = uz * ux * (1 - cos) - uy * sin
    R[2][1] = uz * uy * (1 - cos) + ux * sin
    R[2][2] = cos + uz**2 * (1 - cos)
    return R


def combine_motions(motion1, motion2):
    """
    Combine two sets of motion parameters
    :param motion1: rotation matrix and translation vector 1
    :param motion2: rotation matrix and translation vector 2
    :return: combined set of motion parameters
    """
    R1, t1 = motion1[0], motion1[1]
    R2, t2 = motion2[0], motion2[1]
    t1 /= np.linag.norm(t1)
    t2 /= np.linag.norm(t2)
    t = t1/2 + t2/2
    angle1 = calculate_rotation_angle(R1)
    angle2 = calculate_rotation_angle(R2)
    axis1 = calculate_rotation_axis(R1)
    axis2 = calculate_rotation_axis(R2)
    angle = (angle1 + angle2)/2
    axis = (axis1 + axis2)/2
    R = rotation_matrix_from_axis_and_angle(axis, angle)
    motion = (R, t)
    return motion


# noinspection SpellCheckingInspection
class Sequencer:
    SEQ_NUM = 1
    BUF_SIZ = 2
    HOR_CELLS = 40
    VER_CELLS = 20
    FOV_V = 55
    FOV_H = 94.4
    WINDOW = 15
    CAMERA_ANGLE_X = 0
    CAMERA_ANGLE_Y = 0
    CAMERA_ANGLE_Z = -15
    T_VCF_CCF = np.array([-0.5, -1, 0])
    RADIUS = 10000
    PITCH_THRESHOLD = 5
    ROLL_THRESHOLD = 5
    ANGLE_THRESHOLD = 5
    T_THRESHOLD = 0.2

    def __init__(self):
        self.y2 = 0
        self.buffer = None
        self.principal_point = None
        self.horizon = None
        self.frames = None
        self.channels = None
        self.width = None
        self.height = None
        self.ego_car = None
        self.mask = None
        self.black = None
        self.color = 0
        self.orb = cv2.ORB_create(8000)
        self.a = 0
        self.b = 0
        self.c = 0
        self.K = np.zeros((3, 3))
        self.K_extra = np.zeros((3, 4))

        self.position = np.array([0., 0., 0.])
        self.transformation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.positions = np.array([self.position])
        self.angles = np.array([[0, 0, 0]])
        self.rotation = None
        self.translation = None
        self.normal = None

        self.pos_plot = plt.figure()
        self.ax = self.pos_plot.add_subplot(111, projection='3d', label='test')
        self.angles_plot, self.angles_ax = plt.subplots()

        self.pointsFifo = collections.deque(maxlen=self.BUF_SIZ)
        self.groundPointsFifo = collections.deque(maxlen=self.BUF_SIZ)
        self.generalPointsFifo = collections.deque(maxlen=self.BUF_SIZ)

        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        for base, dirs, files in os.walk('./Videos'):
            for directories in dirs:
                self.SEQ_NUM += 1
        self.imageFifo = collections.deque(maxlen=self.BUF_SIZ)
        self.folder = './Videos/sequence_'

    def show_menu(self):
        """
        Console interaction menu to chose what to do
        """
        print("What do you want to do?")
        print("1: Read video and convert to image folder")
        print("2: Read images in buffer")
        print("Q: quit")
        task = input()
        if task == '1':
            self.read_video()
        elif task == '2':
            sequence = input("Which sequence do you want to use? ")
            while not path.exists('./Videos/sequence_' + str(sequence).zfill(3)):
                sequence = input("Give an existing sequence: ")
            self.read_images(sequence)
        elif task != 'Q' and task != 'q':
            print("Choose one of the options please.")
            self.show_menu()

    def read_video(self):
        """
        Read video of given filename and save each frame as separate image in a folder
        """
        file = input("Give filename: ")
        while file != "":
            if not os.path.exists('./' + file):
                print(file + " does not exist. Try again.")
            else:
                os.mkdir("./Videos/sequence_" + str(self.SEQ_NUM).zfill(3))
                video_to_images.video_to_images(file, self.SEQ_NUM)
                self.SEQ_NUM += 1
            file = input("Give filename: ")
        self.show_menu()

    def process_points(self):
        """
        Filter keypoints (for either homography and essential matrix calculation)
        and add to the propper points FIFO
        :return: None
        """
        points, descriptors = self.dispose(self.pointsFifo[0], 0)
        self.groundPointsFifo.appendleft((points, descriptors))

        points, descriptors = self.dispose(self.pointsFifo[0], 1)
        self.generalPointsFifo.appendleft((points, descriptors))

    def next_image(self, sequence, index, title):
        """
        Add next image to buffer and FIFO
        Detect and filter keypoints
        Show processed image with keypoints
        :param sequence: sequence number
        :param index: index of current image
        :param title: title for window
        :return: index of next frame (-1 when end of file is reached)
        """
        eof = self.add_next_image(sequence, index)
        if eof:
            return -1
        self.detect(self.imageFifo[0], 0)
        self.process_points()

        out = self.show_image(self.groundPointsFifo[0][0], self.imageFifo[0])
        cv2.imshow(title, out)
        index += 1
        return index

    def previous_image(self, sequence, index, title):
        """
        Add previous image to buffer and FIFO
        Detect and filter keypoints
        Show processed image with keypoints
        :param sequence: sequence number
        :param index: index of current image
        :param title: title for window
        :return: index of previous frame
        """
        self.buffer = cv2.imread(
            self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5)
            + '.jpg')
        self.imageFifo.appendleft(self.process_image(self.buffer))
        self.process_points()
        out = self.show_image(self.groundPointsFifo[0][0], self.imageFifo[0])
        cv2.imshow(title, out)
        index -= 1
        return index

    def jump_forward(self, sequence, index, title):
        """
        Jump forward in sequence.
        Add that image to buffer and FIFO
        Detect and filter keypoints
        Show processed image with keypoints
        :param sequence: sequence number
        :param index: index of current image
        :param title: title for window
        :return: index of next frame (-1 when jump would be out of bounds)
        """
        jump = input("How many frames do you want to jump? ")
        while not jump.isnumeric():
            jump = input("Give (positive) number please: ")
        jump = int(jump)
        start = index - self.BUF_SIZ + jump
        if not os.path.exists(
                self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(start).zfill(5) + '.jpg'):
            print("Jump not possible, end of file would be reached")
            return -1
        index += jump
        self.imageFifo.clear()
        self.fill_fifo(sequence, start, index)
        self.process_points()
        out = self.show_image(self.groundPointsFifo[0][0], self.imageFifo[0])
        cv2.imshow(title, out)
        return index

    def jump_backward(self, sequence, index, title):
        """
        Jump backward in sequence.
        Add that image to buffer and FIFO
        Detect and filter keypoints
        Show processed image with keypoints
        :param sequence: sequence number
        :param index: index of current image
        :param title: title for window
        :return: index of next frame (-1 if jump would be out of bounds)
        """
        jump = input("How many frames do you want to jump backwards? ")
        while not jump.isnumeric():
            jump = input("Give (positive) number please: ")
        jump = int(jump)
        start = index - self.BUF_SIZ - jump
        if start < 1:
            print("Jump not possible, too far back.")
            return -1
        index -= jump
        self.imageFifo.clear()
        self.fill_fifo(sequence, start, index)
        self.process_points()
        out = self.show_image(self.groundPointsFifo[0][0], self.imageFifo[0])
        cv2.imshow(title, out)
        return index

    def play(self, sequence, index, title):
        """
        Continuously add next image to buffer and FIFO
        Detect and filter keypoints
        Show processed image with keypoints
        :param sequence: sequence number
        :param index: index of current image
        :param title: title for window
        :return: end of file reached, index of next frame
        """
        eof = False
        key = None
        while not key == ord(' ') and not eof:
            eof = self.add_next_image(sequence, index)
            self.detect(self.imageFifo[0])
            self.process_points()
            out = self.show_image(self.groundPointsFifo[0][0], self.imageFifo[0])
            cv2.imshow(title, out)
            index += 1
            key = cv2.waitKey(1)

        return eof, index

    def select_and_show_keypoints(self, index, title):
        """
        Manually select keypoints
        Show the selected keypoints
        :param index: index of current frame
        :param title: title for window
        :return: 
        """
        # Manually select keypoints
        good_matches = self.select_keypoints(index, title)
        image = cv2.cvtColor(self.imageFifo[0], cv2.COLOR_GRAY2BGR)
        image = reduce_contrast(image)
        image[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black
        color = (0, 255, 0)
        for match in good_matches:
            image = cv2.circle(image, match[0], radius=6, color=color, thickness=3)
            image = cv2.circle(image, match[1], radius=6, color=color, thickness=3)
            image = cv2.line(image, match[0], match[1], color=color, thickness=2)
        cv2.imshow(title, image)

    def calculate_motion(self, sequence, index, title):
        """
        Continuously calculate motion based on homography and essential matrix (if possible)
        :param sequence: sequence number
        :param index: index of current image
        :param title: title for window
        :return: end of file reached, index of next frame
        """
        eof = False
        key = None
        while not key == ord('q') and not eof:
            eof = self.add_next_image(sequence, index)
            exists_h, H, points1, points2 = self.find_homography(index, title)
            if not exists_h:
                h_motion = None
                print('No homography')
            else:
                h_motion = self.decompose_homography(H, points1, points2)

            exists_e, E = self.find_essential(index, title)
            if not exists_e:
                e_motion = None
                print("no e-matrix")
            else:
                e_motion = self.decompose_essential(E)
            # If both are None: use previous motion
            # If only one is None: use other motion
            # If both are not None: combine if possible, otherwise choose best motion
            if h_motion is None:
                if e_motion is None:
                    if self.rotation is None or self.translation is None:
                        print('No motion analysis possible')
                        break
                    motion = (self.rotation, self.translation, self.normal)
                else:
                    motion = e_motion
            else:
                if e_motion is None:
                    motion = h_motion
                else:
                    if self.check_motion_compliance(h_motion, e_motion):
                        motion = combine_motions(h_motion, e_motion)
                    else:
                        motion = self.find_best_motion(h_motion, e_motion)

            self.update_transformations(motion[0], motion[1])

            out = self.show_image(None, self.imageFifo[0])
            self.plot_angles()
            self.plot_positions()
            cv2.imshow(title, out)
            index += 1
            key = cv2.waitKey(0)
        return eof, index

    def show_motion_vectors(self, good_matches, index, title, epipolar=False):
        """
        Show keypoints in current and next frame with their resulting motion vectors
        If epipolar is True, show epipolar lines based on essential matrix between keypoint matches
        :param good_matches: list of matches of keypoints
        :param index: index of current image
        :param title: title for window
        :param epipolar: Show epipolar lines or not
        :return: image with motion vectors (and epipolar lines)
        """
        image = cv2.cvtColor(self.imageFifo[0], cv2.COLOR_GRAY2BGR)
        image = reduce_contrast(image)
        image[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black
        points1 = np.zeros((len(good_matches), 2))
        points2 = np.zeros((len(good_matches), 2))
        i = 0
        for match in good_matches:
            point1 = (match[0], match[1])
            point2 = (match[2], match[3])
            points1[i][0] = point1[0]
            points1[i][1] = point1[1]
            points2[i][0] = point2[0]
            points2[i][1] = point2[1]
            i = i + 1
            image = cv2.circle(image, point1, radius=6, color=(255, 0, 0), thickness=3)
            image = cv2.circle(image, point2, radius=6, color=(255, 0, 0), thickness=3)
            image = cv2.line(image, point1, point2, color=(255, 0, 0), thickness=3)
        if epipolar:
            found, essential = self.find_essential(index, title)
            if not found:
                print("Not found")
            else:
                lines = self.predict_epilines(essential, points1)
                for line in lines:
                    pts = points_on_line(line[0][0], line[0][1], line[0][2], self.width)
                    image = cv2.line(image, pts[0], pts[1], color=(255, 0, 0), thickness=1)
        return image

    def find_horizon_line(self, index, title):
        """
        Calculate homography to find horizon line
        Height can be changed manually to find horizon line at right height
        :param index: index of current image
        :param title: title for window
        :return:
        """
        exists, H, points1, points2 = self.find_homography(index, title)
        if not exists:
            print("No homography found")
            return
        motion = self.decompose_homography(H, points1, points2)
        if motion is None:
            print("No good motion parameters")
            return
        else:
            print(motion[0], '\n', motion[1], '\n', motion[2], '\n', np.linalg.norm(motion[1]))
        k = self.estimate_horizon(motion[2])
        while k != ord('q'):
            if k == ord('n'):
                self.T_VCF_CCF[1] *= -1
            elif k == ord('d'):
                self.T_VCF_CCF[1] /= 2
            else:
                self.T_VCF_CCF[1] *= 2
            k = self.estimate_horizon(motion[2])
        print(int(self.T_VCF_CCF[1]))

    def read_images(self, sequence):
        """
        Read and interact with the images of a sequence, possibilities are the following:
            n: read the next image in the sequence and display it
            p: read previous image in the sequence and display it
            j: Jump forward in sequence (amount entered via console) and display image
            b: Jump backward in sequence (amount entered via console) and display image
            space bar: continuously step forward in the sequence and display image (until space bar is pressed again)
            t: Manually step through keypoints of current image and matches with next image. Press G to save match,
               press any key to discard match
            h: Calculates the homography and essential matrix between the two current frames, decomposes them and
               updates the motion. Shows the motion parameters in a graph
            l: Load saved keypoint matches of current image with next image and display
            u: Test function
            v: Calculate homography, decompose it and predict horizon based on normal vector
            q: Quit program
            :param sequence: Number of sequence from which to load images
            :return:
        """
        bufindex = 0
        index = self.BUF_SIZ + 1
        self.read_info(sequence)

        self.folder = './Videos/sequence_' + str(sequence).zfill(3)
        self.create_mask(sequence)
        self.fill_fifo(sequence, 1, self.BUF_SIZ + 1)

        title = 'Sequence' + str(sequence).zfill(3)
        self.process_points()

        out = self.show_image(self.groundPointsFifo[0][0], self.imageFifo[0])
        cv2.imshow(title, out)
        cv2.setWindowTitle(title, title + ' Frame 1')
        eof = False
        while not eof:
            cv2.setWindowTitle(title, title + ' Frame ' + str(index + bufindex - self.BUF_SIZ))
            key = cv2.waitKey(0)
            if key == ord('n'):
                index = self.next_image(sequence, index, title)
                if index == -1:
                    continue
            elif key == ord('p'):
                if index <= 0:
                    print("Beginning of sequence.")
                    continue
                index = self.previous_image(sequence, index, title)
            elif key == ord('j'):
                i = self.jump_forward(sequence, index, title)
                if i > 0:
                    index = i
            elif key == ord('b'):
                i = self.jump_backward(sequence, index, title)
                if i > 0:
                    index = i
            elif key == ord(' '):
                eof, index = self.play(sequence, index, title)
            elif key == ord('t'):
                self.select_and_show_keypoints(index, title)
            elif key == ord('h'):
                eof, index = self.calculate_motion(sequence, index, title)
            elif key == ord('l'):
                # Load stored matches and show motion vectors
                # Additionally, show epipolar lines
                good_matches = self.load_points(index, title, 0)
                image = self.show_motion_vectors(good_matches, index, title, True)
                cv2.imshow(title, image)
            elif key == ord('u'):
                # Test birds eye view
                height = 1
                angle = 1
                cont = True
                while cont:
                    cont, k = self.test(index, title, height, angle)
                    if k == ord('a'):
                        angle *= 2
                    else:
                        height *= 2
                print(height/2)
            elif key == ord('v'):
                self.find_horizon_line(index, title)
            elif key == ord('q'):
                eof = True
            else:
                continue
        cv2.destroyAllWindows()
        return

    def find_best_motion(self, motion1, motion2):
        """
        Finds motion that is closest to last known motion
        :param motion1:
        :param motion2:
        :return: best motion parameters
        """
        if self.rotation is None or self.translation is None:
            # If no motion is found yet, use motion by essential matrix
            return motion2
        previous_motion = (self.rotation, self.translation)
        if self.check_motion_compliance(motion1, previous_motion):
            return motion1
        elif self.check_motion_compliance(motion2, previous_motion):
            return motion2
        else:
            anglediff1 = calculate_rotation_angle(motion1[0]) - calculate_rotation_angle(self.rotation)
            anglediff2 = calculate_rotation_angle(motion2[0]) - calculate_rotation_angle(self.rotation)
            if anglediff1 < anglediff2:
                return motion1
            else:
                return motion2

    def check_motion_compliance(self, motion1, motion2):
        """
        Check whether motions are compatible (that means they don't differ too much)
        :param motion1: rotation matrix and translation vector 1
        :param motion2: rotation matrix and translation vector 2
        :return: boolean that states whether motions comply
        """
        R1, t1 = motion1[0], motion1[1]
        R2, t2 = motion2[0], motion2[1]
        print(R1, '\n', R2)
        R = np.matmul(R1, np.transpose(R2))
        rot_angle = calculate_rotation_angle(R)
        if math.degrees(rot_angle) > self.ANGLE_THRESHOLD:
            return False
        t1 /= np.linalg.norm(t1)
        t2 /= np.linalg.norm(t2)
        t = t1 - t2
        return np.linalg.norm(t) < self.T_THRESHOLD

    def create_mask(self, sequence):
        """
        Read mask of ego-car and store it for later use (depending on the use of color images the mask is different)
        :param sequence:
        :return:
        """
        ego_car = cv2.imread(self.folder + '/SEQ' + str(sequence).zfill(3) + 'BW.jpg')
        (T, ego_car) = cv2.threshold(ego_car, 127, 255, cv2.THRESH_BINARY)
        self.mask = cv2.cvtColor(ego_car, cv2.COLOR_BGR2GRAY)
        self.ego_car = ego_car / 255
        if self.color == 0:
            self.black = [0]
        else:
            self.black = [0, 0, 0]
        self.ego_car = self.ego_car[self.horizon:self.height, 0:self.width]
        self.mask = self.mask[self.horizon:self.height, 0:self.width]

    def fill_fifo(self, sequence, start, stop):
        """
        Fill FIFO with images from index "start" until index "stop" after being processed, keep last new image in buffer
        :param sequence:
        :param start:
        :param stop:
        :return:
        """

        for i in range(start, stop):
            # check whether end of file is reached
            if not os.path.exists(
                    self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(i).zfill(5) + '.jpg'):
                return
            self.buffer = cv2.imread(self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(i)).zfill(5)
                                     + '.jpg')
            self.imageFifo.appendleft(self.process_image(self.buffer))
            self.detect(self.imageFifo[0], 0)

    def add_next_image(self, sequence, index):
        """
        Add next image to FIFO after preprocessing and store original in buffer
        :param sequence:
        :param index:
        :return:
        """
        # check whether end of file is reached
        if not os.path.exists(
                self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg'):
            self.imageFifo.pop()
            if len(self.imageFifo) == 0:
                return True
        # add image to the FIFO
        self.buffer = cv2.imread(
            self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5)
            + '.jpg')
        self.imageFifo.appendleft(self.process_image(self.buffer))
        return False

    def check_angles(self, rotation):
        """
        Check whether angles of roation matrix are smaller than threshold values
        :param rotation: Rotation matrix (3x3)
        :return: boolean that denotes whether angles are both smaller than threshold
        """
        angles = rotation_matrix_to_euler_angles(rotation)
        return angles[0] < self.PITCH_THRESHOLD and angles[2] < self.ROLL_THRESHOLD

    def decompose_homography(self, homography, points1, points2):
        """
        Decompose homography and select the right motion parameters
        :param homography: Homography to decompose
        :param points1: points in image 1
        :param points2: points in image 2
        :return: The best possible set of motion parameters (if any)
        """
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(homography, self.K)
        motions = np.array([])
        best = 0
        for i in range(retval):
            # Check whether motion parameters could be right and how many points are "good points"
            possible, goodpoints = check_possibility(rotations[i], translations[i], normals[i], points2)
            if possible and goodpoints > best:
                motions = np.append(motions, i)
                best = goodpoints
        if len(motions == 1):
            i = int(motions[0])
            motion = (rotations[i], translations[i], normals[i])
            return motion
        else:
            for i in motions:
                if self.check_angles(rotations[i]):
                    motion = (rotations[i], translations[i], normals[i])
                    return motion
        return None

    def decompose_essential(self, essential):
        """
        Decompose essential matrix and select the right motion parameters
        :param essential: Essential matrix to decompose
        :param points1: points in image 1
        :param points2: points in image 2
        :return: The best possible set of motion parameters (if any)
        """
        R1, R2, t = cv2.decomposeEssentialMat(essential)
        # Make sure rotation is positive in z-direction
        if t[2] < 0:
            t = np.negative(t)
        # Not sure how to find right rotation matrix
        R = None
        if self.check_angles(R1):
            R = R1
        elif self.check_angles(R2):
            R = R2
        motion = (R, t)
        return motion

    def find_homography(self, index, title):
        """
        Estimate homography between frames index and index + 1
        :param index:
        :param title:
        :return: Boolean: homography found or not, homography, points used to find homography in second frame
        """
        # Calculate homography and decompose it
        good_matches = self.load_points(index, title, 0)
        points1 = np.zeros((len(good_matches), 2), dtype=np.int32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.int32)
        i = 0
        for match in good_matches:
            points1[i, :] = self.cropped_to_original((match[0], match[1]))
            points2[i, :] = self.cropped_to_original((match[2], match[3]))
            i += 1
        if i < 4:
            print("Not enough matches")
            return False, None, None, None
        # print("Calculating Homography:")
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        return True, H, points1, points2

    def find_essential(self, index, title):
        """
        Estimate essential matrix between frames index and index + 1
        :param index:
        :param title:
        :return: Boolean: essential matrix found or not, E matrix
        """
        # Calculate homography and decompose it
        good_matches = self.load_points(index, title, 1)
        points1 = np.zeros((len(good_matches), 2), dtype=np.int32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.int32)
        i = 0
        for match in good_matches:
            points1[i, :] = self.cropped_to_original((match[0], match[1]))
            points2[i, :] = self.cropped_to_original((match[2], match[3]))
            i += 1
        if i < 5:
            print("Not enough matches")
            return False, None
        E, mask = cv2.findEssentialMat(points1, points2, self.K, cv2.RANSAC)
        return True, E

    def update_transformations(self, rotation, translation, normal=None):
        """
        Updates transformation of car with current motion
        :param rotation: rotation matrix (3x3)
        :param translation: translation vector (3x1)
        :param normal: normal vector (3x1)
        :return:
        """
        self.rotation = rotation
        self.translation = translation
        if normal is not None:
            self.normal = normal
        t = translation / np.linalg.norm(translation)
        t = self.ccf_to_wcf(t)
        self.position += t
        T = form_transformation_matrix(rotation, t)
        self.transformation = np.matmul(self.transformation, T)
        current_position = np.array([self.transformation[0][3],
                                     self.transformation[1][3],
                                     self.transformation[2][3]])
        self.positions = np.append(self.positions, [self.position], axis=0)
        angles = rotation_matrix_to_euler_angles(rotation)
        self.angles = np.append(self.angles, [angles], axis=0)

    def integrate_rotation(self, rotation):
        old_rot = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                old_rot = self.transformation[i][i]
        old_rot = np.transpose(old_rot)
        return np.multiply(old_rot, rotation)

    def load_points(self, index, title, point_type=0):
        """
        Load keypoints and matches between frame on index and index + 1
        If not keypoints stored yet, let user select them manually
        :param index: current index
        :param title: path name of current sequence
        :param point_type: 0 is points on plane, 1 is points everywhere
        :return: Matches between keypoints (loaded or selected)
        """
        # point_type 0 is points on a plane->homography, type 1 is all points->essential matrix
        if point_type == 0:
            filename = self.folder + '/points/homography/IMG' + str(int(index - 2)).zfill(5) + "-" + str(
                int(index - 1)).zfill(
                5) + '.txt'
        elif point_type == 1:
            filename = self.folder + '/points/essential/IMG' + str(int(index - 2)).zfill(5) + "-" + str(
                int(index - 1)).zfill(
                5) + '.txt'
        else:
            return
        if not path.exists(filename):
            good_matches = self.select_keypoints(index, title, point_type)
            good_matches = good_matches.reshape(len(good_matches), 4)
        else:
            good_matches = np.loadtxt(filename, dtype='int32', delimiter=',')
        return good_matches

    def test(self, index, title, height, angle):
        """
        Test function
        :param index:
        :param title:
        :param height:
        :param angle:
        :return:
        """
        # k = ''
        # while k != ord('q'):
        #     angle = input("Give angle: ")
        #     angle = int(angle)
        #     rotated = rotate_image(self.buffer, angle)
        #     cv2.imshow("rotated", rotated)
        #     k = cv2.waitKey(0)
        # cv2.destroyWindow("rotated")
        # temp = self.vcf_to_ccf([10, 15, 0])
        # print(self.ccf_to_vcf(temp))
        # birds_eye_view(self.buffer, self.K, self.CAMERA_ANGLE_X, self.CAMERA_ANGLE_Z)
        exists, H, points1, points2 = self.find_homography(index, title)
        if not exists:
            print("No homography found")
            return
        motion = self.decompose_homography(H, points1, points2)
        if motion is None:
            print("No good motion parameters")
            return False, None
        bev = self.create_birds_eye_view(motion[2], height, angle)
        cv2.imshow("birds eye view", bev)
        k = cv2.waitKey(0)
        cv2.destroyWindow('birds eye view')
        if k == ord('q'):
            return False, None
        return True, k

    def create_birds_eye_view(self, normal, height=2000, angle=1):
        """
        Compose bird's eye view of current frame by warping perspective
        :param normal: plane normal vector (3x1)
        :param height: (height of camera)
        :param angle: (
        :return: bird's eye view
        """
        # rotation = make_rotation_matrix(self.CAMERA_ANGLE_X, 0, self.CAMERA_ANGLE_Z, radians=False)
        # translation = np.zeros((3, 3))
        # translation[0][0] = 1
        # translation[1][1] = 1
        # translation[2][2] = 2000
        # rotation = np.matmul(rotation, translation)
        # h1 = np.matmul(self.K, rotation)
        # h1_inv = np.linalg.inv(h1)
        # works, h2, points = self.find_homography(index, title)
        # h12 = np.matmul(h2, h1_inv)
        # bev = cv2.warpPerspective(self.buffer, h12, (self.buffer.shape[1], self.buffer.shape[0]))
        rotation = rotation_matrix_from_vectors(np.array([0, 1, 0]), -normal)
        translation = np.zeros((3, 3))
        translation[0][0] = 1
        translation[1][1] = 1
        translation[2][2] = angle
        rotation = np.matmul(rotation, translation)
        k_inv = np.linalg.inv(self.K)
        # rot = make_rotation_matrix(self.CAMERA_ANGLE_X, 0, self.CAMERA_ANGLE_Z, radians=False)
        # rot = np.linalg.inv(rot)
        # h1 = np.matmul(k_inv, rot)
        h1 = k_inv
        h2 = np.matmul(self.K, rotation)
        h12 = np.matmul(h2, h1)
        nt = np.matmul(np.array([[500], [height], [500]]), normal.T)
        H = np.add(rotation, nt)
        bev = cv2.warpPerspective(self.buffer, H, (self.buffer.shape[1], self.buffer.shape[0]))

        return bev

    def select_keypoints(self, index, title, point_type=-1):
        """
        Manually walk through keypoint matches between frame on index and index + 1.
        Select good matches by pressing 'G', discard match by pressing 'B'
        Store these matches for later reuse
        :param index: index of current frame
        :param title: Title of window for showing purposes
        :param point_type: 0 for points on ground plane, 1 for all points, -1 to ask for user input.
        :return: good matches
        """
        if point_type == -1:
            point_type = input("\n0. Homography\n1. Essential")
            while point_type != '0' and point_type != '1':
                point_type = input("\n0. Homography\n1. Essential")
        point_type = int(point_type)
        if point_type == 0:
            filename = self.folder + '/points/homography/IMG' + str(int(index - 2)).zfill(5) + '-' \
                       + str(int(index - 1)).zfill(5) + '.txt '
            kp1, des1 = self.groundPointsFifo[0][0], self.groundPointsFifo[0][1]
        else:
            filename = self.folder + '/points/essential/IMG' + str(int(index - 2)).zfill(5) + '-' \
                       + str(int(index - 1)).zfill(5) + '.txt '
            kp1, des1 = self.generalPointsFifo[0][0], self.generalPointsFifo[0][1]

        kp2, des2 = self.pointsFifo[1][0], self.pointsFifo[1][1]
        des1 = des1.astype('uint8')
        matches = self.matcher.match(des1, des2)
        image = cv2.cvtColor(self.imageFifo[0], cv2.COLOR_GRAY2BGR)
        image[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black
        img1 = self.imageFifo[0]
        img2 = self.imageFifo[1]
        good_matches = []
        for match in matches:
            point1 = [int(kp1[match.queryIdx][0]), int(kp1[match.queryIdx][1])]
            point2 = [int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1])]
            color = (255, 0, 0)
            image = cv2.cvtColor(self.imageFifo[0], cv2.COLOR_GRAY2BGR)
            image[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black
            image = cv2.circle(image, point1, radius=6, color=color, thickness=3)
            image = cv2.circle(image, point2, radius=6, color=color, thickness=3)
            image = cv2.line(image, point1, point2, color=color, thickness=2)
            # image = cv2.resize(image, (int(self.width / 1.25), int((self.height - self.horizon) / 1.25)))
            cv2.imshow(title, image)
            # cv2.waitKey()
            cross_color1 = (255, 255, 255)
            cross_color2 = (255, 255, 255)

            patch1 = self.create_patch(img1, point1)
            patch2 = self.create_patch(img2, point2)
            patches = np.hstack((patch1, patch2))

            patches = cv2.resize(patches, (20 * self.WINDOW, 10 * self.WINDOW))
            cv2.imshow("patches", patches)
            k = cv2.waitKey(0)
            if k == ord('g'):
                good_matches.append([point1, point2])
            elif k == ord('q'):
                break
        cv2.destroyWindow("patches")
        np.savetxt(filename, np.array(good_matches).reshape(len(good_matches), 4), fmt='%i', delimiter=",")
        return np.array(good_matches)

    def create_patch(self, image, point):
        """
        Create small patch of image around point
        :param image: Image to take patches from
        :param point: Point around which patch should be made
        :return: patch
        """
        cross_color = (255, 255, 255)
        patch = copy(image[point[1] - self.WINDOW // 2:point[1] + self.WINDOW // 2,
                point[0] - self.WINDOW // 2: point[0] + self.WINDOW // 2])
        black = np.average(patch) < 128
        if not black:
            cross_color = (0, 0, 0)
        patch = cv2.line(patch, (self.WINDOW // 2, 0), (self.WINDOW // 2, self.WINDOW), cross_color)
        patch = cv2.line(patch, (0, self.WINDOW // 2), (self.WINDOW, self.WINDOW // 2), cross_color)

        return patch

    def dispose(self, kp_des, point_type):
        """
        Remove excess of keypoints by:
            Bucketing keypoints (see function "bucket()")
            Removing points above the horizon (if point_type is 0)
        :param kp_des: array of keypoints with their respective descriptor
        :param point_type: 0 for points on plane, 1 for all points
        :return: remaining keypoints and descriptors
        """
        if point_type == 1:
            check_horizon = False
        else:
            check_horizon = True
        kps = np.array([])
        descs = np.array([])
        points = kp_des[0]
        des = kp_des[1]
        des_len = len(des[0])
        cells = [[[0, 0]] * self.HOR_CELLS] * self.VER_CELLS
        descriptors = [[[-1] * des_len] * self.HOR_CELLS] * self.VER_CELLS
        filled = 0
        full = self.HOR_CELLS * self.VER_CELLS
        b = self.width / self.HOR_CELLS
        h = (self.height - self.horizon) / self.VER_CELLS
        for i in range(len(points)):
            point = points[i].pt
            x, y = int(point[0] / b), int(point[1] / h)
            in_mask = (self.ego_car[int(point[1])][int(point[0])] == [0., 0., 0.]).all()
            above = check_horizon and (self.a * int(point[0]) + self.b * int(point[1]) + self.c) < 0
            if in_mask or above:
                continue
            if all(v == 0 for v in cells[y][x]):
                cells[y][x] = np.array(point)
                kps = np.append(kps, point)
                descs = np.append(descs, np.array(des[i]))
                descriptors[y][x] = np.array(des[i])
                filled += 1
            if filled == full:
                break
        # cells = np.array(cells).reshape((self.HOR_CELLS * self.VER_CELLS, 2))
        # cells = cells[cells != [0, 0]].reshape(-1, 2)
        # descriptors = np.array(descriptors).reshape((self.HOR_CELLS * self.VER_CELLS, -1))
        # descriptors = descriptors[descriptors != [-1] * des_len].reshape(-1, des_len)
        keypoints = kps.reshape(-1, 2)
        descriptors = descs.reshape(-1, des_len)
        return keypoints, descriptors

    def dont_dispose(self, kp_des, t):
        kps = np.array([])
        descs = np.array([])
        points = kp_des[0]
        des = kp_des[1]
        des_len = len(des[0])
        for i in range(len(points)):
            point = points[i].pt
            kps = np.append(kps, point)
            descs = np.append(descs, np.array(des[i]))

        keypoints = kps.reshape(-1, 2)
        descriptors = descs.reshape(-1, des_len)
        return keypoints, descriptors

    def find_point_on_epipolar_line(self, old_point, old_descriptor, line, new_points, new_descriptors):
        a, b, c = line
        filter_array = []
        for point in new_points:
            filter_array.append(point_in_range(point, a, b, c))
        possible_points = new_points[filter_array]
        possible_descriptors = new_descriptors[filter_array].astype('uint8')
        matches = self.matcher.match(np.array([old_descriptor]), possible_descriptors)
        print(matches)

        # match = (old_point, new_point)
        # return matches

    def read_info(self, sequence):
        """
        Read info of info file:
            dimensions of footage
            channels
            amount of frmaes
            horizon: height of horizon on the left (empirically chosen)
            y2: height of horizon on the right side (empirically chosen)
        Calculate based on these:
            principal point
            a, b and c: parameters of horizon line
            Camera matrix K
        :param sequence: Sequence number
        :return:
        """
        info = open('Videos/sequence_' + str(sequence).zfill(3) + '/info.txt', 'r')
        info.readline()
        self.height = int(info.readline().split(' ')[-1])
        self.width = int(info.readline().split(' ')[-1])
        self.channels = int(info.readline().split(' ')[-1])
        self.frames = int(info.readline().split(' ')[-1])
        self.horizon = int(info.readline().split(' ')[-1])
        self.principal_point = (int(self.width / 2), int(self.height / 2))
        self.y2 = int(info.readline().split(' ')[-1])
        self.a = self.horizon - self.y2
        self.b = self.width
        self.c = 0
        self.K[0][0] = self.width / (2 * tan(math.radians(self.FOV_H / 2)))
        self.K[1][1] = self.height / (2 * tan(math.radians(self.FOV_V / 2)))
        self.K[0][2] = self.principal_point[0]
        self.K[1][2] = self.principal_point[1]
        self.K[2][2] = 1
        unity = np.zeros((3, 4))
        for i in range(3):
            unity[i][i] = 1
        self.K_extra = np.matmul(self.K, unity)

    def detect(self, image, pos=0):
        """
        Detect keypoints and compute descriptors in image
        :param image: Image to detect keypoints on
        :param pos: position in the FIFO (going forwards or backwards)
        :return:
        """
        kp, des = self.orb.detectAndCompute(image, self.mask)
        if pos == 0:
            self.pointsFifo.appendleft((kp, des))
        else:
            self.pointsFifo.append((kp, des))

    def process_image(self, image):
        """
        Pre process image:
            convert to grayscale
            crop image, removing everything above "horizon"
        :param image:
        :return:
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.circle(image, self.principal_point, radius=5, color=(255, 0, 0), thickness=3)
        image = image[self.horizon:self.height, 0:self.width]
        return image

    def show_image(self, points, image):
        """
        Prepare image to be shown
        :param points: keypoints to be drawn on image
        :param image: image to be shown
        :return: Image ready to be shown
        """
        img = reduce_contrast(image)
        # converted to BGR so keypoints can be shown in color
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black
        if points is None:
            return img
        for i in range(len(points)):
            point = (int(points[i][0]), int(points[i][1]))
            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            img = cv2.circle(img, point, radius=6, color=color, thickness=3)
            # horizon line
            # img = cv2.line(img, (0, 0), (self.width, self.y2 - self.horizon), color=(0, 0, 0), thickness=2)
        # # Draw bucket grid
        # w = int(self.width/self.HOR_CELLS)
        # h = int(self.height/self.VER_CELLS)
        # for i in range(1, self.HOR_CELLS):
        #     img = cv2.line(img, (i*w, 0), (i*w, self.height), color=(0, 0, 0), thickness=1)
        # for i in range(1, self.HOR_CELLS):
        #     img = cv2.line(img, (0, i*h), (self.width, i*h), color=(0, 0, 0), thickness=1)
        return img

    def cropped_to_original(self, coordinate):
        """
        Transfer image coordinates from original to cropped
        :param coordinate: coordinate to transfer
        :return: coordinate in original image
        """
        return [coordinate[0], coordinate[1] + self.horizon]

    def plot_positions(self):
        """
        Plot position in 3D graph
        :return:
        """
        xdata = np.array([])
        ydata = np.array([])
        zdata = np.array([])
        for pos in self.positions:
            xdata = np.append(xdata, pos[0])
            ydata = np.append(ydata, pos[1])
            zdata = np.append(zdata, pos[2])
        self.ax.plot(xdata, ydata, zdata, color='b')
        self.pos_plot.canvas.draw()
        self.pos_plot.show()

    def plot_angles(self):
        """
        Plot euler angles of rotation
        :return:
        """
        angles = np.transpose(self.angles)
        phis = angles[0]
        psis = angles[1]
        thetas = angles[2]
        x = np.array(range(len(phis)))
        self.angles_ax.plot(x, phis, color='r')
        self.angles_ax.plot(x, psis, color='g')
        self.angles_ax.plot(x, thetas, color='b')
        self.angles_plot.canvas.draw()
        self.angles_plot.show()

    def vcf_to_ccf(self, vector):
        """
        Convert vector from vehicle coordinate frame to camera coordinate frame
        :param vector: vector (3X1)
        :return: converted vector
        """
        r_vcf_to_ccf = make_rotation_matrix(self.CAMERA_ANGLE_X, self.CAMERA_ANGLE_Y, self.CAMERA_ANGLE_Z, radians=False)
        ccf = np.matmul(r_vcf_to_ccf, vector)
        ccf = np.add(ccf, self.T_VCF_CCF)
        return ccf

    def ccf_to_vcf(self, vector):
        """
        Convert vector from camera coordinate frame to vehicle coordinate frame
        :param vector: vector (3X1)
        :return: converted vector
        """
        r_ccf_to_vcf = np.transpose(make_rotation_matrix(self.CAMERA_ANGLE_X,
                                                         self.CAMERA_ANGLE_Y,
                                                         self.CAMERA_ANGLE_Z, radians=False))
        t_vcf_ccf = [[-self.T_VCF_CCF[0]], [-self.T_VCF_CCF[1]], [-self.T_VCF_CCF[2]]]
        vcf = np.add(vector, t_vcf_ccf)
        vcf = np.matmul(r_ccf_to_vcf, vcf)
        return vcf

    def vcf_to_wcf(self, vector):
        """
        Convert vector from vehicle coordinate frame to world coordinate frame
        :param vector: vector (3X1)
        :return: converted vector
        """
        # invert transformation of car so far
        t = invert_transform_matrix(self.transformation)
        vector = np.append(np.array(vector), [1])
        wcf = np.matmul(t, vector)
        # rotate 180° around z-axis
        r = make_rotation_matrix(0, 0, 180, radians=False)
        wcf = np.matmul(r, wcf[:-1])
        return wcf

    def wcf_to_vcf(self, vector):
        """
        Convert vector from world coordinate frame to vehicle coordinate frame
        :param vector: vector (3X1)
        :return: converted vector
        """
        # rotate 180° around z-axis
        r = make_rotation_matrix(0, 0, 180, radians=False)
        vcf = np.matmul(r, vector)
        # apply transformation of car so far
        vcf = np.append(vcf, [1])
        vcf = np.matmul(self.transformation, vcf)
        return vcf

    def wcf_to_ccf(self, vector):
        """
        Convert vector from world coordinate frame to camera coordinate frame
        :param vector: vector (3X1)
        :return: converted vector
        """
        vcf = self.wcf_to_vcf(vector)
        ccf = self.vcf_to_ccf(vcf)
        return ccf

    def ccf_to_wcf(self, vector):
        """
        Convert vector from camera coordinate frame to world coordinate frame
        :param vector: vector (3X1)
        :return: converted vector
        """
        vcf = self.ccf_to_vcf(vector)
        wcf = self.vcf_to_wcf(vcf)
        return wcf

    def estimate_horizon(self, normal):
        """
        Estimate horizon line based on plane normal vector
        Show for verifying purposes
        :param normal: Ground plane normal vector
        :return:
        """
        d = -self.T_VCF_CCF[1]
        point1 = point_in_distance(normal, 70, 100000, d)
        point2 = point_in_distance(normal, 60, 100000, d)
        p1 = np.matmul(self.K_extra, point1)
        p2 = np.matmul(self.K_extra, point2)
        p1 /= p1[2]
        p2 /= p2[2]
        p1[1] = p1[1] - self.horizon
        p2[1] = p2[1] - self.horizon
        start = get_horizon_point(p1, p2, 0)
        end = get_horizon_point(p1, p2, self.width)
        image = cv2.cvtColor(self.imageFifo[1], cv2.COLOR_GRAY2BGR)
        image = cv2.line(image, start, end, color=(255, 0, 0), thickness=2)
        cv2.imshow('horizon', image)
        key = cv2.waitKey(0)
        cv2.destroyWindow('horizon')
        return key

    def find_point_under_camera(self, normal):
        """
        Find point under camera on ground plane
        :param normal: Ground plane normal vector
        :return: Point under camera
        """
        d = self.T_VCF_CCF[1]
        return d * normal

    def predict_epilines(self, essential, points):
        """
        Predict the epilines in the next image based on the current essential matrix and the used keypoints.
        An epipolar line is described by its 3 parameters a, b and c of
        :param essential: Essential matrix (3x3)
        :param points: Keypoints in image
        :return: Array of epipolar lines corresponding to the given points
        """
        # F = (K^-1)^T*E*K^-1
        k_inv = np.linalg.inv(self.K)
        F = np.matmul(np.transpose(k_inv), essential)
        F = np.matmul(F, k_inv)
        lines = cv2.computeCorrespondEpilines(points, 1, F)
        return lines

