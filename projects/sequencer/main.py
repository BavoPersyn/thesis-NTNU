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
        buffer(sequence)
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


def buffer(sequence, bufsiz=2):
    nextright = True
    index = bufsiz + 1
    imageQueue = collections.deque(maxlen=bufsiz)
    folder = './Videos/sequence_' + str(sequence).zfill(3)
    for i in range(1, bufsiz + 1):
        image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(i)).zfill(5) + '.jpg')
        imageQueue.append(image)
    cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])
    key = cv2.waitKey(0)
    eof = False
    while not eof:
        if key == ord('n'):
            imageQueue.popleft()
            image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg')
            # print('Reading ' + folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg')
            imageQueue.append(image)
            cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])
            index += 1
            nextright = True
            key = cv2.waitKey()
        elif key == ord('p'):
            if nextright:
                index -= (bufsiz + 1)
            if index < 1:
                index = 1
                print("Beginning of sequence.")
                key = cv2.waitKey()
                continue
            imageQueue.pop()
            image = cv2.imread(folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg')
            # print('Reading ' + folder + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(index)).zfill(5) + '.jpg')
            imageQueue.appendleft(image)
            cv2.imshow('Sequence' + str(sequence).zfill(3), imageQueue[0])
            index -= 1
            nextright = False
            key = cv2.waitKey()
        elif key == ord('q'):
            eof = True
        else:
            key = cv2.waitKey()
    return


show_menu(SEQ_NUM)
