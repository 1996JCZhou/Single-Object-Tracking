def calculate_IOU(target_bb, predict_bb):

    # Coordinations of the target bounding box.
    lt_x0, lt_y0, rb_x0, rb_y0 = target_bb

    # Coordinations of the prediction bounding box.
    lt_x1, lt_y1, rb_x1, rb_y1 = predict_bb

    # Compute the radius of each bounding box.
    radius_target_x = (rb_x0 - lt_x0) / 2.0
    redius_predict_x = (rb_x1 - lt_x1) / 2.0

    # Compute the center of each bounding box.
    center_target_x = (rb_x0 + lt_x0) / 2.0
    center_predict_x = (rb_x1 + lt_x1) / 2.0

    # Check if the two bounding boxes intersect.
    if abs(center_target_x - center_predict_x) > (radius_target_x + redius_predict_x):
	    return 0.
    else:
        # Sort the coordinations of x and y of the two bounding boxes.
        sorted_x = sorted([lt_x0, rb_x0, lt_x1, rb_x1])
        sorted_y = sorted([lt_y0, rb_y0, lt_y1, rb_y1])

        # Compute the intersection area of the two bounding boxes.
        intersection_area = (sorted_x[2] - sorted_x[1]) * (sorted_y[2] - sorted_y[1])

        # Compute the union area of the two bounding boxes.
        union_area = (rb_x0 - lt_x0) * (rb_y0 - lt_y0) + (rb_x1 - lt_x1) * (rb_y1 - lt_y1) - intersection_area

        # Compute the IoU of the two bounding boxes.
        IoU = intersection_area / (union_area + 1e-7)
        return IoU


if __name__ == '__main__':
    IoU = calculate_IOU([100, 100, 200, 200], [110, 110, 210, 210])
    print(IoU)
