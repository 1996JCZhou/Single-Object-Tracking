import numpy as np
import cv2   as cv

"""Load video."""
cap = cv.VideoCapture("D:\\Python Spyder\\Objekterkennung\\Meanschift\\slow_traffic_small.mp4")

"""Define the target model (ROI) positions as a window to track
   from the first frame of the video by using the 'Read_Pixel_Position.py' file.
"""
# Take the first frame of the video.
ret, frame = cap.read() # 'frame.shape' = (360, 640, 3).
# 'ret=True' : Data read. / 'ret=False': Data not read.
# cv.imwrite("D:\\Python Spyder\\Objekterkennung\\Meanschift\\the_first_frame.png", frame)

# Define the target model (ROI) positions as a window to track.
x, y, w, h = 299, 186, 90, 31
track_window = (x, y, w, h)
# Set up the ROI in the first frame of the video for tracking.
roi = frame[y:y+h, x:x+w] # 'roi.shape' = (22, 22, 3).

"""Calculate the estimation of the color representation of the target model
   (histogram with kernel function, probability of the color in the target model).
"""
# Transform the ROI of the current frame as a BGR image into the HSV color model.
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV) # 'hsv_roi.shape' = (22, 22, 3).
# Low light values are filtered using the 'cv.inRange()' function
# to avoid false values due to low light.
# 'cv.inRange()': Takes in a three-channel HSV image as imput and outputs an one-channel grey value image.
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.))) # 'mask.shape' = (22, 22).
# 'cv2.calcHist([images], [channel], mask, histSize, ranges)':
# 1. '[images]': Input image.
# 2. For histogram, only Hue (the "0-te" channel) of the HSV color model image is considered here.
# 3. For histogram, only the masked region (ROI) will be taken into consideration.
# 4. '[histSize]': Number of value intervalls that are required to display the histogram.
#    For example, every pixel Hue value is considered in the histogram, then we have 180 bins.
# 5. 'ranges': Range of pixel values (the histogram bin boundaries).
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180]) # 'roi_hist.shape' = (180, 1).
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
# 'sum(roi_hist)' = 184 < 484.
# That means 300 pixels have been filtered according to 'mask' along the 'S' and 'V' axises.

"""Set the termination criteria."""
# The type of termination criteria is 'cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT', which means
# the convergence can be ended when one of the following two conditions is reached:
# the threshold of two pixel positions 'epsilon' and the maximum number of iterations.
# The maximum number of iterations is set to be 10, and the threshold of two pixel positions 'epsilon' is set to be 1.
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    """Run every frame of the video until the video ends ('ret' = False)."""
    ret, frame = cap.read()
    if ret == True:
        """Apply 'Back Projection' of the target model histogram to the current frame."""
        # Transform the current frame as a BGR image into the HSV color model.
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # 'calcBackProject()': Record how well the pixels of a given image fit the distribution of pixels in a histogram model.
        # Calculate the histogram model of a feature and then use it to find this feature in an (bigger) image.
        # For example: If you have a histogram of flesh color (a Hue-Saturation histogram),
        # then you can use it to find flesh color areas in an image.
        # 1. '[images]': Input image.
        # 2. '[channels]': The list of channels used to compute the back projection.
        #    For histogram, only Hue (the "0-te" channel) of the HSV color model image is considered here.
        # 3. 'hist': Input histogram.
        # 4. 'ranges': Range of pixel values (the histogram bin boundaries).
        # 5. 'scale': Scaling factor. Here we set this parameter to 1,
        #    because the values of the target model histogram are already scaled to '0-255'.
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # 1. At each location (x, y) of the test image from the selected channel,
        #    the function collects the pixel value and finds the corresponding target model histogram bin ('which bin').
        # 2. The function stores the bin value of this bin from the target model histogram at pixel location (x, y) in a new image 'dst'.

        """Apply meanshift to find the target candidate."""
        # 'meanShift()':
        # 1. 'dst': Result of the 'Back Projection'.
        # 2. 'track_window': The esimated target candidate window in the previous frame.
        # 3. 'term_crit': Termination criteria.
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        """Draw the estimated result / target candidate on the image."""
        x, y, w, h = track_window
        target_candidate = cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.imshow('Target candidate', target_candidate)

        """Press the 'Esc' key to immediately exit the program."""
        # The ASCII code value corresponding to the same key on the keyboard in different situations (e.q. when the key "NumLock" is activated)
        # is not necessarily the same, and does not necessarily have only 8 bits, but the last 8 bits must be the same.
        # In order to avoid this situation, quote &0xff, to get the last 8 bits of the ASCII value of the pressed key
        # to determine what the key is.
        k = cv.waitKey(30) & 0xff
        if k == ord("q"):
            break
    else:
        break
