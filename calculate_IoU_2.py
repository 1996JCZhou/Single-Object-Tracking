import cv2
import numpy as np


def calculate_IoU(RecA, RecB):
    """Calculate the IoU.

    Args:
        RecA (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].
        RecB (List): [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].

    Returns:
        IoU (Float): Intersection over union.
    """

    # The x coordinate of the left top pixel point of the right bounding box.
    xA = max(RecA[0], RecB[0])
    # The x coordinate of the right bottom pixel point of the left bounding box.
    xB = min(RecA[2], RecB[2])
    # The y coordinate of the left top pixel point of the bottom bounding box.
    yA = max(RecA[1], RecB[1])
    # The y coordinate of the right bottom pixel point of the top bounding box.
    yB = min(RecA[3], RecB[3])

    intersection_x = xB - xA if xB - xA >= 1 else 0
    intersection_y = yB - yA if yB - yA >= 1 else 0
    intersection_area = intersection_x * intersection_y

    RecA_Area = (RecA[2] - RecA[0]) * (RecA[3] - RecA[1])
    RecB_Area = (RecB[2] - RecB[0]) * (RecB[3] - RecB[1])

    iou = intersection_area / float(RecA_Area + RecB_Area - intersection_area)

    return iou


def cal_iou(box1, box2):
    """
    Calculate the IoU of two bounding boxes.

    Args:
        box1 (List): [top left x coordinate, top left y coordinate, bottom right x coordinate, bottom right y coordinate].
        box2 (List): [top left x coordinate, top left y coordinate, bottom right x coordinate, bottom right y coordinate].

    Returns:
        IoU (Float): Intersection over union.
    """

    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]

    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # The x coordinate of the top left pixel point of the right bounding box.
    xmin = max(x1min, x2min)
    # The x coordinate of the bottom right pixel point of the left bounding box.
    xmax = min(x1max, x2max)
    # The y coordinate of the top left pixel point of the bottom bounding box.
    ymin = max(y1min, y2min)
    # The y coordinate of the bottom right pixel point of the top bounding box.
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    IoU = intersection / (s1 + s2 - intersection)

    return IoU


img = np.zeros((512, 512, 3), np.uint8)
img.fill(255)

RecA = [100, 100, 200, 200]
RecB = [120, 120, 220, 220]

cv2.rectangle(img, (RecA[0], RecA[1]), (RecA[2], RecA[3]), (0, 255, 0), 1)
cv2.rectangle(img, (RecB[0], RecB[1]), (RecB[2], RecB[3]), (255, 0, 0), 1)

IoU = cal_iou(RecA, RecB)
# IoU = calculate_IoU(RecA, RecB)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, "IOU = %.2f" %IoU, (300, 500), font, 0.8, (0, 0, 0), 1)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
