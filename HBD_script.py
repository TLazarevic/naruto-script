import dlib
import cv2
import imutils
from imutils import face_utils
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("preview")

sharingan = cv2.imread("Sharingan_Triple.png")
headband = cv2.imread("headband2.jpg")
rasengan = cv2.imread("shuriken.png", -1)
rasengan = cv2.resize(rasengan, (100, 100))

sharingan_shape = sharingan.shape
headband_shape = headband.shape
rasengan_shape = rasengan.shape
shuriken_location = 6
shuriken_speed = 20

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def euclidianDistance(a,b):
    return np.math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

while cap.isOpened():
    rval, img = cap.read()

    # img = imutils.resize(img, width=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(img, 0)
    for (i, rect) in enumerate(rects):
        dots = predictor(gray, rect)
        dots = face_utils.shape_to_np(dots)

        right_eye_ul = dots[37]
        right_eye_ur = dots[38]
        right_eye_ll = dots[41]
        right_eye_lr = dots[40]
        right_eye = img[right_eye_ur[1]:right_eye_lr[1], right_eye_ul[0]:right_eye_ur[0]]
        r_eye_shape = right_eye.shape

        left_eye_ul = dots[43]
        left_eye_ur = dots[44]
        left_eye_ll = dots[47]
        left_eye_lr = dots[46]
        left_eye = img[left_eye_ur[1]:left_eye_lr[1], left_eye_ul[0]:left_eye_ur[0]]
        l_eye_shape = left_eye.shape

        scale_factor = euclidianDistance(dots[1], dots[15])
        forehead_ul = [dots[0][0] + 7, dots[19][1] - int(scale_factor/5)]
        forehead_ur = [dots[16][0] - 7, dots[24][1] - int(scale_factor/5)]
        forehead_ll = [dots[0][0] + 7, dots[19][1]]
        forehead_lr = [dots[16][0] - 7, dots[24][1]]
        forehead = img[forehead_ur[1]:forehead_lr[1], forehead_ul[0]:forehead_ur[0]]
        forehead_shape = forehead.shape

        r_eye_points = np.float32([[right_eye_ul[0], right_eye_ul[1]], [right_eye_ur[0], right_eye_ur[1]],
                                   [right_eye_ll[0], right_eye_ll[1]]])
        l_eye_points = np.float32([[left_eye_ul[0], left_eye_ul[1]], [left_eye_ur[0], left_eye_ur[1]],
                                   [left_eye_ll[0], left_eye_ll[1]]])
        forehead_points = np.float32([[forehead_ul[0], forehead_ul[1]], [forehead_ur[0], forehead_ur[1]],
                                      [forehead_ll[0], forehead_ll[1]]])
        sharingan_points = np.float32([[0, 0], [sharingan_shape[0], 0], [0, sharingan_shape[1]]])
        headband_points = np.float32([[0, 0], [headband_shape[0], 0], [0, headband_shape[1]]])

        M = cv2.getAffineTransform(sharingan_points, r_eye_points)
        dst = cv2.warpAffine(sharingan, M, (img.shape[1], img.shape[0]))
        M2 = cv2.getAffineTransform(sharingan_points, l_eye_points)
        dst2 = cv2.warpAffine(sharingan, M2, (img.shape[1], img.shape[0]))
        M3 = cv2.getAffineTransform(headband_points, forehead_points)
        dst3 = cv2.warpAffine(headband, M3, (img.shape[1], img.shape[0]))

        resized_headband = cv2.resize(headband, (int(forehead.shape[1]), int(forehead.shape[0])))
        # Now create a mask of headband and create its inverse mask also
        roi = forehead
        img2gray = cv2.cvtColor(resized_headband, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of headband in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of headband from logo image.
        img2_fg = cv2.bitwise_and(resized_headband, resized_headband, mask=mask)
        # Put headband in ROI and modify the main image
        distance = cv2.add(img1_bg, img2_fg)
        img[forehead_ur[1]:forehead_lr[1], forehead_ul[0]:forehead_ur[0]] = resized_headband

        final = cv2.add(img, dst)
        final = cv2.add(final, dst2)

        if img.shape[1]> rasengan.shape[1]+shuriken_location + shuriken_speed:
            shuriken_location = shuriken_location + shuriken_speed
        else:
            shuriken_location = 6
        final = overlay_transparent(final, rasengan, shuriken_location, int(shuriken_location/3))

        cv2.line(final, (dots[1][0] + 20, dots[1][1] + 10), (dots[31][0] - 20, dots[31][1] - 20), (0, 0, 0), 2)
        cv2.line(final, (dots[2][0] + 20, dots[2][1] + 10), (dots[31][0] - 20, dots[31][1] - 10), (0, 0, 0), 2)
        cv2.line(final, (dots[3][0] + 20, dots[3][1] + 10), (dots[31][0] - 20, dots[31][1]), (0, 0, 0), 2)

        cv2.line(final, (dots[15][0] - 20, dots[15][1] + 10), (dots[35][0] + 20, dots[35][1] - 20), (0, 0, 0), 2)
        cv2.line(final, (dots[14][0] - 20, dots[14][1] + 10), (dots[35][0] + 20, dots[35][1] - 10), (0, 0, 0), 2)
        cv2.line(final, (dots[13][0] - 20, dots[13][1] + 10), (dots[35][0] + 20, dots[35][1]), (0, 0, 0), 2)

        cv2.imshow("preview", final)
        key = cv2.waitKey(10)
        if key == 27:  # exit on ESC
            cap.release()
            cv2.destroyWindow("preview")
