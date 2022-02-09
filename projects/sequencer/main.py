import cv2
import collections
N = 2
scale_percent = 25 # percent of original size

def readimage(number):
    image = cv2.imread('Images/Foto' + str(i) + '.jpg', -1)
    dim = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    imageQueue.append(image)

imageQueue = collections.deque(maxlen=N)

i = 1

while i <= 50:
    readimage(i)
    cv2.imshow("foto", imageQueue[-1])
    key = cv2.waitKey(0)
    goodkey = False
    while not goodkey:
        if key == ord('n'):
            goodkey = True
            i += 1
        elif key == ord('t'):
            goodkey = True
            i += 10
        else:
            key = cv2.waitKey(0)

cv2.destroyAllWindows()

