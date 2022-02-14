import cv2


def video_to_images(filename, sequence=1, to_grayscale=False):
    cap = cv2.VideoCapture(filename)
    while not cap.isOpened():
        cap = cv2.VideoCapture(filename)
        cv2.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    prog_100 = 0
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            if to_grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('video', gray)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cv2.imwrite('Videos/sequence_' + str(sequence).zfill(3) + '/SEQ'
                        + str(sequence).zfill(3) + 'IMG' + str(int(pos_frame)).zfill(5) + '.jpg', frame)
            # print('Videos/sequence_' + str(sequence).zfill(3) + '/SEQ' + str(sequence).zfill(3) + 'IMG' + str(int(
            # pos_frame)).zfill(5))
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        if cv2.waitKey(10) == ord('q'):
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            print(filename + ' read and converted.')
            break
        else:
            progress = cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT)
            prog_10 = int(progress * 10)
            progress = int(progress * 100)
            if progress != prog_100:
                prog_100 = progress
                print("Progress: [" + prog_10*'=' + '>' + (10-prog_10)*' ' + '] ' + str(progress) + '%', end="\r")
    write_info(sequence, cap.get(cv2.CAP_PROP_FRAME_COUNT))


def write_info(sequence, frames):
    first = cv2.imread('Videos/sequence_' + str(sequence).zfill(3) + '/SEQ'
                       + str(sequence).zfill(3) + 'IMG' + str(1).zfill(5) + '.jpg')
    f = open('Videos/sequence_' + str(sequence).zfill(3) + '/info.txt', 'w')
    f.write('Sequence: ' + str(sequence).zfill(3) + '\n')
    height, width, channels = first.shape
    f.write('Height: ' + str(height) + '\nWidth: ' + str(width) + '\nChannels: ' + str(channels) + '\n')
    f.write('Frames: ' + str(int(frames)))
