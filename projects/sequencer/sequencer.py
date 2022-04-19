import cv2
import collections
import os
import video_to_images
import os.path
from os import path
import numpy as np
import random as rand


class Sequencer:
    SEQ_NUM = 1
    BUF_SIZ = 2
    HOR_CELLS = 40
    VER_CELLS = 20
    FOV_V = 55
    FOV_H = 94.4
    WINDOW = 15

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
        self.pointsFifo = collections.deque(maxlen=self.BUF_SIZ)
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        for base, dirs, files in os.walk('./Videos'):
            for directories in dirs:
                self.SEQ_NUM += 1
        self.imageFifo = collections.deque(maxlen=self.BUF_SIZ)
        self.folder = './Videos/sequence_'

    def show_menu(self):
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

    def create_mask(self, sequence):
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
        # fill FIFO with the next images after being processed, keep last new image in buffer
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

    def bucket(self, points):
        cells = [[[0, 0]] * self.HOR_CELLS] * self.VER_CELLS
        filled = 0
        full = self.HOR_CELLS * self.VER_CELLS
        points1, points2 = points[0], points[1]
        b = self.width / self.HOR_CELLS
        h = (self.height - self.horizon) / self.VER_CELLS
        for point in points1:
            x, y = int(point[0] / b), int(point[1] / h)
            if all(v == 0 for v in cells[y][x]):
                cells[y][x] = np.array(point)
                filled += 1
            if filled == full:
                break
        cells = np.array(cells).reshape((self.HOR_CELLS * self.VER_CELLS, 2))
        return cells

    def read_images(self, sequence):
        bufindex = 0
        index = self.BUF_SIZ + 1
        self.read_info(sequence)

        self.folder = './Videos/sequence_' + str(sequence).zfill(3)
        self.create_mask(sequence)
        self.fill_fifo(sequence, 1, self.BUF_SIZ + 1)

        title = 'Sequence' + str(sequence).zfill(3)
        points, descriptors = self.dispose(self.pointsFifo[0])
        out = self.show_image((points, points), self.imageFifo[0])
        cv2.imshow(title, out)
        cv2.setWindowTitle(title, title + ' Frame 1')
        eof = False
        while not eof:
            cv2.setWindowTitle(title, title + ' Frame ' + str(index + bufindex - self.BUF_SIZ))
            key = cv2.waitKey(0)
            if key == ord('n'):
                eof = self.add_next_image(sequence, index)
                if eof:
                    continue
                # points = self.detect_and_match()
                self.detect(self.imageFifo[0], 0)
                points, descriptors = self.dispose(self.pointsFifo[0])
                out = self.show_image((points, points), self.imageFifo[0])
                cv2.imshow(title, out)
                index += 1
            elif key == ord('p'):
                if index <= 0:
                    print("Beginning of sequence.")
                    continue
                # add image to the FIFO
                self.buffer = cv2.imread(
                    self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5)
                    + '.jpg')
                self.imageFifo.appendleft(self.process_image(self.buffer))
                # points = self.detect_and_match()
                points, descriptors = self.dispose(self.pointsFifo[0])
                out = self.show_image((points, points), self.imageFifo[0])
                cv2.imshow(title, out)
                index -= 1
            elif key == ord('j'):
                jump = input("How many frames do you want to jump? ")
                while not jump.isnumeric():
                    jump = input("Give (positive) number please: ")
                jump = int(jump)
                start = index - self.BUF_SIZ + jump
                if not os.path.exists(
                        self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(start).zfill(5) + '.jpg'):
                    print("Jump not possible, end of file would be reached")
                    continue
                index += jump
                self.imageFifo.clear()
                self.fill_fifo(sequence, start, index)
                points, descriptors = self.dispose(self.pointsFifo[0])
                out = self.show_image((points, points), self.imageFifo[0])
                cv2.imshow(title, out)
            elif key == ord('b'):
                jump = input("How many frames do you want to jump backwards? ")
                while not jump.isnumeric():
                    jump = input("Give (positive) number please: ")
                jump = int(jump)
                start = index - self.BUF_SIZ - jump
                if start < 1:
                    print("Jump not possible, too far back.")
                    continue
                index -= jump
                self.imageFifo.clear()
                self.fill_fifo(sequence, start, index)
                points, descriptors = self.dispose(self.pointsFifo[0])
                out = self.show_image((points, points), self.imageFifo[0])
                cv2.imshow(title, out)
            elif key == ord(' '):
                key = None
                # cv2.setWindowTitle(title, title + ' playing.')
                while not key == ord(' ') and not eof:
                    eof = self.add_next_image(sequence, index)
                    self.detect(self.imageFifo[0])
                    points, descriptors = self.dispose(self.pointsFifo[0])
                    out = self.show_image((points, points), self.imageFifo[0])
                    cv2.imshow(title, out)
                    index += 1
                    key = cv2.waitKey(1)
            elif key == ord('s'):
                file = open(self.folder + "/points/IMG" + str(int(index - 2)).zfill(5) +
                            "-" + str(int(index - 1)).zfill(5) + ".txt", "a")
                (points1, points2) = self.detect_and_match()
                im1 = self.reduce_contrast(self.imageFifo[-2])
                im2 = self.reduce_contrast(self.imageFifo[-1])
                goodpoints = []
                for i in range(0, len(points1)):
                    image = np.vstack((im1, im2))
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    p1 = (int(points1[i][0]), int(points1[i][1]))
                    p2 = (int(points2[i][0]), int(points2[i][1]) + self.height - self.horizon)
                    cv2.circle(image, p1, radius=6, color=(255, 0, 0), thickness=3)
                    cv2.circle(image, p2, radius=6, color=(255, 0, 0), thickness=3)
                    cv2.imshow("test", image)
                    k = cv2.waitKey(0)
                    if k == ord('q'):
                        break
                    elif k == ord('g'):
                        file.write(str(p1) + str(p2) + "\n")
                    else:
                        continue
                file.close()
                cv2.destroyWindow("test")
            elif key == ord('t'):
                good_matches = self.select_keypoints(index, title)
                image = cv2.cvtColor(self.imageFifo[0], cv2.COLOR_GRAY2BGR)
                image[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black
                color = (0, 255, 0)
                for match in good_matches:
                    image = cv2.circle(image, match[0], radius=6, color=color, thickness=3)
                    image = cv2.circle(image, match[1], radius=6, color=color, thickness=3)
                    image = cv2.line(image, match[0], match[1], color=color, thickness=2)
                cv2.imshow(title, image)
            elif key == ord('h'):
                filename = self.folder + '/points/IMG' + str(int(index - 2)).zfill(5) + "-" + str(int(index - 1)).zfill(5) + '.txt'
                if not path.exists(filename):
                    good_matches = self.select_keypoints(index, title)
                    good_matches = good_matches.reshape(len(good_matches), 4)
                else:
                    good_matches = np.loadtxt(filename, dtype='int32', delimiter=',')
                points1 = np.zeros((len(good_matches), 2), dtype=np.int32)
                points2 = np.zeros((len(good_matches), 2), dtype=np.int32)
                i = 0
                for match in good_matches:
                    point1 = (match[0], match[1])
                    point2 = (match[2], match[3])
                    points1[i, :] = point1
                    points2[i, :] = point2
                    i += 1
                print("calculating homography:")
                h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
                print(h)
            elif key == ord('q'):
                eof = True
            else:
                continue
        cv2.destroyAllWindows()
        return

    def select_keypoints(self, index, title):
        filename = self.folder + '/points/IMG' + str(int(index - 2)).zfill(5) + '-' \
                   + str(int(index - 1)).zfill(5) + '.txt '
        kp1, des1 = self.dispose(self.pointsFifo[0])
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
            cv2.imshow(title, image)
            # cv2.waitKey()
            patch1 = img1[point1[1] - self.WINDOW // 2:point1[1] + self.WINDOW // 2,
                     point1[0] - self.WINDOW // 2: point1[0] + self.WINDOW // 2]
            patch2 = img2[point2[1] - self.WINDOW // 2:point2[1] + self.WINDOW // 2,
                     point2[0] - self.WINDOW // 2:point2[0] + self.WINDOW // 2]

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

    def dispose(self, kp_des):
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
            above = (self.a * int(point[0]) + self.b * int(point[1]) + self.c) < 0
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

    def read_info(self, sequence):
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

    def detect(self, image, pos=0):
        kp, des = self.orb.detectAndCompute(image, self.mask)
        if pos == 0:
            self.pointsFifo.appendleft((kp, des))
        else:
            self.pointsFifo.append((kp, des))

    def detect_and_match(self):
        # convert images in buffer to grayscale
        img1 = self.imageFifo[-1]
        img2 = self.imageFifo[-2]
        # compute keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, self.mask)
        kp2, des2 = self.orb.detectAndCompute(img2, self.mask)
        # match keypoints and sort them
        matches = self.matcher.match(des1, des2, None)
        matches = sorted(matches, key=lambda x: x.distance)
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        good_matches = []
        j = 0
        for match in matches:

            p1 = (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1]))
            p2 = (int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1]))
            # Check if point is part above the horizon
            above = (self.a * p1[0] + self.b * p1[1] + self.c) < 0 or (self.a * p2[0] + self.b * p2[1] + self.c) < 0
            if above:
                # at least one of the keypoints is part of the ego-car or above the horizon, this match will be ignored
                continue
            good_matches.append(match)
            points1[j, :] = kp1[match.queryIdx].pt
            points2[j, :] = kp2[match.trainIdx].pt
            j += 1
        points1 = np.reshape(points1[points1 != [0., 0.]], (-1, 2))
        points2 = np.reshape(points2[points2 != [0., 0.]], (-1, 2))
        if j <= 4:
            print("Not enough points to calculate Homography.")
        else:
            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
            # print("Estimated homography : \n", h)
        # show the best 20 matches
        # out = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None)
        # out = cv2.pyrDown(out)
        # cv2.imshow("Matches", out)

        return [points1, points2]

    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.circle(image, self.principal_point, radius=5, color=(255, 0, 0), thickness=3)
        image = image[self.horizon:self.height, 0:self.width]
        return image

    def show_image(self, points, image):
        img = self.reduce_contrast(image)
        # converted to BGR so keypoints can be shown in color
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[np.where((self.ego_car <= [0, 0, 0]).all(axis=2))] = self.black

        # Show only the first 20, these are the best matches, bad matches make it unclear
        for i in range(len(points[0])):
            point1 = (int(points[0][i][0]), int(points[0][i][1]))
            point2 = (int(points[1][i][0]), int(points[1][i][1]))
            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            img = cv2.circle(img, point2, radius=6, color=color, thickness=3)
            # img = cv2.circle(img, point2, radius=3, color=color, thickness=2)
            # img = cv2.line(img, point1, point2, color=color, thickness=2)
            # horizon line
            # img = cv2.line(img, (0, 0), (self.width, self.y2 - self.horizon), color=(0, 0, 0), thickness=2)
        return img

    def reduce_contrast(self, image):
        mapper = np.vectorize(lambda x: (x * 127) // 255 + 128)
        image = mapper(image).astype(np.uint8)
        return image
