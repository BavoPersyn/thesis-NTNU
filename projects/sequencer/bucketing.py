import cv2

image = cv2.imread('Videos/sequence_001/SEQ001IMG00332.jpg', -1)
cv2.imshow('test', image)
cv2.waitKey(0)
height = 1080
width = 1920
buckets = 10
orb = cv2.ORB_create(50)
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

images = []
h = height//buckets
w = width//buckets

for i in range(buckets):
    images.append([])
    for j in range(buckets):
        # print(str(i*h)+':'+str(i*h+h), str(j*w) + ':' + str(j*w+w))
        up = 0
        down = 0
        left = 0
        right = 0
        images[i].append(image[i*h:i*h+h, j*w:j*w+w])

