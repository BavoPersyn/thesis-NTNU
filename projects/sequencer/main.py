import cv2
import collections
import os
import video_to_images
import os.path
from os import path

SEQ_NUM = 1

for base, dirs, files in os.walk('./Videos'):
    for directories in dirs:
        SEQ_NUM += 1


def show_menu(sequence_number):
    print("What do you want to do?")
    print("1: Read video and convert to image folder")
    print("2: Read images in buffer")
    print("Q: quit")
    task = input()
    if task == '1':
        read_video(sequence_number)
    elif task == '2':
        sequence = input("Which sequence do you want to use? ")
        while not path.exists('./Videos/sequence_' + str(sequence).zfill(3)):
            sequence = input("Give an existing sequence: ")
        buffer(sequence, downsampling=True)
    elif task != 'Q' and task != 'q':
        print("Choose one of the options please.")
        show_menu(sequence_number)


def read_video(sequence_number):
    file = input("Give filename: ")

    while file != "":
        if not os.path.exists('./' + file):
            print(file + " does not exist. Try again.")
        else:
            os.mkdir("./Videos/sequence_" + str(sequence_number).zfill(3))
            video_to_images.video_to_images(file, sequence_number)
            sequence_number += 1
        file = input("Give filename: ")
    show_menu(sequence_number)


def buffer(sequence, bufsiz=2, color=0, downsampling=False):
    bufindex = 0
    index = bufsiz + 1
    height, width, channels, frames, horizon = read_info(sequence)
    imageQueue = collections.deque(maxlen=bufsiz)
    folder = './Videos/sequence_' + str(sequence).zfill(3)
    for i in range(1, bufsiz + 1):
        image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(i)).zfill(5) + '.jpg', color)
        image = image[horizon:height, 0:width]
        if downsampling:
            image = cv2.pyrDown(image)
        imageQueue.append(image)
    title = 'Sequence' + str(sequence).zfill(3)
    cv2.imshow(title, imageQueue[0])
    cv2.setWindowTitle(title, title + ' Frame 1')
    eof = False
    while not eof:
        cv2.setWindowTitle(title, title + ' Frame ' + str(index + bufindex - bufsiz))
        key = cv2.waitKey(0)
        if key == ord('n'):
            if not os.path.exists(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg'):
                bufindex += 1
                if bufindex == bufsiz:
                    eof = True
                else:
                    cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[bufindex])
                continue
            imageQueue.popleft()
            image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg',
                               color)
            image = image[horizon:height, 0:width]
            if downsampling:
                image = cv2.pyrDown(image)
            imageQueue.append(image)
            cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])
            index += 1
        elif key == ord('p'):
            if bufindex > 0:
                bufindex -= 1
                cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[bufindex])
                continue
            previous = index - bufsiz - 1
            if previous < 1:
                print("Beginning of sequence.")
                continue
            imageQueue.pop()
            image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(previous)).zfill(5) + '.jpg',
                               color)
            image = image[horizon:height, 0:width]
            if downsampling:
                image = cv2.pyrDown(image)
            imageQueue.appendleft(image)
            cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])
            index -= 1
        elif key == ord('j'):
            jump = input("How many frames do you want to jump? ")
            while not jump.isnumeric():
                jump = input("Give (positive) number please: ")
            jump = int(jump)
            start = index - bufsiz + jump
            if not os.path.exists(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(start).zfill(5) + '.jpg'):
                print("Jump not possible, end of file would be reached")
                continue
            index += jump
            imageQueue.clear()
            for i in range(start, index):
                if not os.path.exists(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(i).zfill(5) + '.jpg'):
                    continue
                image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(i).zfill(5) + '.jpg', color)
                image = image[horizon:height, 0:width]
                if downsampling:
                    image = cv2.pyrDown(image)
                imageQueue.append(image)
            cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])
        elif key == ord('b'):
            jump = input("How many frames do you want to jump backwards? ")
            while not jump.isnumeric():
                jump = input("Give (positive) number please: ")
            jump = int(jump)
            start = index - bufsiz - jump
            if start < 1:
                print("Jump not possible, too far back.")
                continue
            index -= jump
            imageQueue.clear()
            for i in range(start, index):
                image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(i).zfill(5) + '.jpg', color)
                image = image[horizon:height, 0:width]
                if downsampling:
                    image = cv2.pyrDown(image)
                imageQueue.append(image)
            cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])

        elif key == ord('q'):
            eof = True
        else:
            continue
    return


def read_info(sequence):
    info = open('Videos/sequence_' + str(sequence).zfill(3) + '/info.txt', 'r')
    info.readline()
    height = int(info.readline().split(' ')[-1])
    width = int(info.readline().split(' ')[-1])
    channels = int(info.readline().split(' ')[-1])
    frames = int(info.readline().split(' ')[-1])
    horizon = int(info.readline().split(' ')[-1])
    return height, width, channels, frames, horizon


show_menu(SEQ_NUM)
