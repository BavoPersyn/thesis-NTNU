import cv2
import collections
import os
import video_to_images
import os.path
from os import path
import numpy as np



class Sequencer:

    SEQ_NUM = 1
    BUFSIZ = 2

    def __init__(self):
        self.principal_point = None
        self.horizon = None
        self.frames = None
        self.channels = None
        self.width = None
        self.height = None
        self.mask = None
        self.black = None
        self.color = 0
        for base, dirs, files in os.walk('./Videos'):
            for directories in dirs:
                self.SEQ_NUM += 1
        self.imageQueue = collections.deque(maxlen=self.BUFSIZ)
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
            self.buffer(sequence)
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

    def fill_buffer(self, sequence, start, stop):
        for i in range(start, stop):
            if not os.path.exists(
                    self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(i).zfill(5) + '.jpg'):
                return
            image = image = cv2.imread(self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(i)).zfill(5)
                                       + '.jpg')
            self.imageQueue.append(image)

    def add_next_image(self, sequence, index, bufindex):
        # check whether end of file is reached
        if not os.path.exists(
                self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg'):
            bufindex += 1
            if bufindex == self.BUFSIZ:
                return True, bufindex
            else:
                cv2.imshow('Sequence' + str(sequence).zfill(3), self.imageQueue[bufindex])
            return False, bufindex
        # add image to the right of the buffer
        self.imageQueue.popleft()
        image = image = cv2.imread(self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5)
                                   + '.jpg')
        self.imageQueue.append(image)
        return False, bufindex

    def add_previous_image(self, sequence, index, bufindex):
        if bufindex > 0:
            bufindex -= 1
            cv2.imshow('Sequence' + str(sequence).zfill(3), self.imageQueue[bufindex])
            return bufindex
        previous = index - self.BUFSIZ - 1
        if previous < 1:
            print("Beginning of sequence.")
            return bufindex
        # add image tot the left of the buffer
        self.imageQueue.pop()
        image = image = cv2.imread(self.folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5)
                                   + '.jpg')
        self.imageQueue.appendleft(image)
        return bufindex

    def buffer(self, sequence):
        bufindex = 0
        index = self.BUFSIZ + 1
        self.read_info(sequence)

        self.folder = './Videos/sequence_' + str(sequence).zfill(3)
        self.create_mask(sequence)

        self.fill_buffer(sequence, 1, self.BUFSIZ + 1)

        title = 'Sequence' + str(sequence).zfill(3)
        cv2.imshow(title, self.process_image(self.imageQueue[0]))
        cv2.setWindowTitle(title, title + ' Frame 1')
        eof = False
        while not eof:
            cv2.setWindowTitle(title, title + ' Frame ' + str(index + bufindex - self.BUFSIZ))
            key = cv2.waitKey(0)
            if key == ord('n'):
                eof, bufindex = self.add_next_image(sequence, index, bufindex)
                cv2.imshow('Sequence' + str(sequence).zfill(3), self.process_image(self.imageQueue[0]))
                index += 1
            elif key == ord('p'):
                bufindex = self.add_previous_image(sequence, index, bufindex)
                cv2.imshow('Sequence' + str(sequence).zfill(3), self.process_image(self.imageQueue[0]))
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
                self.imageQueue.clear()
                self.fill_buffer(sequence, start, index)
                cv2.imshow('Sequence' + str(sequence).zfill(3), self.process_image(self.imageQueue[0]))
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
                self.imageQueue.clear()
                self.fill_buffer(sequence, start, index)
                cv2.imshow('Sequence' + str(sequence).zfill(3), self.process_image(self.imageQueue[0]))
            elif key == ord(' '):
                key = None
                cv2.setWindowTitle(title, title + ' playing.')
                while not key == ord(' ') and not eof:
                    eof, bufindex = self.add_next_image(sequence, index, bufindex)
                    cv2.imshow('Sequence' + str(sequence).zfill(3), self.process_image(self.imageQueue[0]))
                    index += 1
                    key = cv2.waitKey(1)
            elif key == ord('q'):
                eof = True
            else:
                continue
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

    def process_image(self, image):
        if self.color == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[np.where((self.mask <= [0, 0, 0]).all(axis=2))] = self.black
        image = cv2.circle(image, self.principal_point, radius=5, color=(255, 0, 0), thickness=3)
        image = image[self.horizon:self.height, 0:self.width]
        image = cv2.pyrDown(image)
        return image