import math

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
    BUFSIZ = 2

    def __init__(self):
        self.y2 = 0
        self.buffer = None
        self.principal_point = None
        self.horizon = None
        self.frames = None
        self.channels = None
        self.width = None
        self.height = None
        self.mask = None
        self.black = None
        self.color = 0
        self.orb = cv2.ORB_create(2000)
        self.a = 0
        self.b = 0
        self.c = 0
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        for base, dirs, files in os.walk('./Videos'):
            for directories in dirs:
                self.SEQ_NUM += 1
        self.imageFifo = collections.deque(maxlen=self.BUFSIZ)
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
        mask = cv2.imread(self.folder + '/SEQ' + str(sequence).zfill(3) + 'BW.jpg')
        (T, mask) = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        self.mask = mask / 255
        if self.color == 0:
            self.black = [0]
        else:
            self.black = [0, 0, 0]

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

    def read_images(self, sequence):
        bufindex = 0
        index = self.BUFSIZ + 1
        self.read_info(sequence)

        self.folder = './Videos/sequence_' + str(sequence).zfill(3)
        self.create_mask(sequence)

        self.fill_fifo(sequence, 1, self.BUFSIZ + 1)

        title = 'Sequence' + str(sequence).zfill(3)
        cv2.imshow(title, self.imageFifo[0])
        self.detect_and_match()
        cv2.setWindowTitle(title, title + ' Frame 1')
        eof = False
        while not eof:
            cv2.setWindowTitle(title, title + ' Frame ' + str(index + bufindex - self.BUFSIZ))
            key = cv2.waitKey(0)
            if key == ord('n'):
                eof = self.add_next_image(sequence, index)
                if eof:
                    continue
                # cv2.imshow(title, self.imageFifo[0])
                points = self.detect_and_match()
                out = self.show_image(points, self.imageFifo[0])
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
                cv2.imshow(title, self.imageFifo[0])
                points = self.detect_and_match()
                out = self.show_image(points, self.imageFifo[0])
                cv2.imshow(title, out)
                index -= 1
            elif key == ord('j'):
                jump = input("How many frames do you want to jump? ")
                while not jump.isnumeric():
                    jump = input("Give (positive) number please: ")
                jump = int(jump)
                start = index - self.BUFSIZ + jump
                if not os.path.exists(
                        self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(start).zfill(5) + '.jpg'):
                    print("Jump not possible, end of file would be reached")
                    continue
                index += jump
                self.imageFifo.clear()
                self.fill_fifo(sequence, start, index)
                points = self.detect_and_match()
                out = self.show_image(points, self.imageFifo[0])
                cv2.imshow(title, out)
            elif key == ord('b'):
                jump = input("How many frames do you want to jump backwards? ")
                while not jump.isnumeric():
                    jump = input("Give (positive) number please: ")
                jump = int(jump)
                start = index - self.BUFSIZ - jump
                if start < 1:
                    print("Jump not possible, too far back.")
                    continue
                index -= jump
                self.imageFifo.clear()
                self.fill_fifo(sequence, start, index)
                points = self.detect_and_match()
                out = self.show_image(points, self.imageFifo[0])
                cv2.imshow(title, out)
            elif key == ord(' '):
                key = None
                # cv2.setWindowTitle(title, title + ' playing.')
                while not key == ord(' ') and not eof:
                    eof = self.add_next_image(sequence, index)
                    points = self.detect_and_match()
                    out = self.show_image(points, self.imageFifo[0])
                    cv2.imshow(title, out)
                    index += 1
                    key = cv2.waitKey(1)
            elif key == ord('q'):
                eof = True
            else:
                continue
        cv2.destroyAllWindows()
        return

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
        self.a = self. horizon - self.y2
        self.b = self.width
        self.c = 0

    def detect_and_match(self):
        # convert images in buffer to grayscale
        img1 = self.imageFifo[-1]
        img2 = self.imageFifo[-2]
        # compute keypoints and descriptors
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
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
            # Check if point is part of ego-car or above the horizon
            in_mask = (self.mask[p1[1]][p1[0]] == [0, 0, 0]).all() or (self.mask[p2[1]][p2[0]] == [0, 0, 0]).all()
            above = (self.a * p1[0] + self.b * p1[1] + self.c) < 0 or (self.a * p2[0] + self.b * p2[1] + self.c) < 0
            if in_mask or above:
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
        image[np.where((self.mask <= [0, 0, 0]).all(axis=2))] = self.black
        # image = cv2.circle(image, self.principal_point, radius=5, color=(255, 0, 0), thickness=3)
        image = image[self.horizon:self.height, 0:self.width]
        return image

    def show_image(self, points, image):
        img = self.reduce_contrast(image)
        # converted to BGR so keypoints can be shown in color
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Show only the first 20, these are the best matches, bad matches make it unclear
        for i in range(len(points[0])):
            point1 = (int(points[0][i][0]), int(points[0][i][1]))
            point2 = (int(points[1][i][0]), int(points[1][i][1]))
            color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            img = cv2.circle(img, point1, radius=3, color=color, thickness=2)
            img = cv2.circle(img, point2, radius=3, color=color, thickness=2)
            img = cv2.line(img, point1, point2, color=color, thickness=1)
        return img

    def reduce_contrast(self, image):
        mapper = np.vectorize(lambda x: (x * 127) // 255 + 128)
        image = mapper(image).astype(np.uint8)
        return image
