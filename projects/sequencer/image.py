import cv2
import collections
import os

N = 2
scale_percent = 25  # percent of original size


def readImage(number):
    image = cv2.imread('Images/Foto' + str(number) + '.jpg', -1)
    dim = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    imageQueue.append(image)


# imageQueue = collections.deque(maxlen=N)
#
# i = 1
#
# while i <= 50:
#     readImage(i)
#     cv2.imshow("foto", imageQueue[-1])
#     key = cv2.waitKey(0)
#     goodKey = False
#     while not goodKey:
#         if key == ord('n'):
#             goodKey = True
#             i += 1
#         elif key == ord('t'):
#             goodKey = True
#             i += 10
#         else:
#             key = cv2.waitKey(0)
#
# cv2.destroyAllWindows()


def readImagesOfDirectory(directory):
    direct = os.fsencode(directory)
    stop = False
    for file in os.listdir(direct):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            imagename = os.path.join(directory, filename)
            image = cv2.imread(imagename, -1)
            cv2.imshow('Image', image)
            key = cv2.waitKey(0)
            goodKey = False
            while not goodKey:
                if key == ord('n'):
                    goodKey = True
                elif key == ord('q'):
                    goodKey = True
                    stop = True
                else:
                    key = cv2.waitKey(0)
            if stop:
                break


readImagesOfDirectory('.\\Images')
