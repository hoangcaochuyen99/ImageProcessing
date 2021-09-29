import cv2

vidcap = cv2.VideoCapture('C:\\PythonFiles\\videopython\\rose.mp4')
success, image = vidcap.read()
count = 0
success = True
while success:
    # save frame as JPEG file
    cv2.imwrite('C:\\PythonFiles\\videopython\\frame%d.jpg' % count, image)
    success, image = vidcap.read()
    print('Read a new frame:', success)
    count += 1