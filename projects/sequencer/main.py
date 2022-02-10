import cv2
import collections
import os
import video_to_images

SEQ_NUM = 1

for base, dirs, files in os.walk('./Videos'):
    for directories in dirs:
        SEQ_NUM += 1


def show_menu(sequence_number):
    print("What do you want to do?")
    print("1: Read video and convert to image folder")
    print("Q: quit")
    task = input()
    if task == '1':
        read_video(sequence_number)
    # elif task == '2':
    #     buffer(sequence_number)
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


def buffer(sequence_number):
    return


show_menu(SEQ_NUM)
