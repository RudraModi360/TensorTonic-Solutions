from math import sqrt
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    anchors = []

    for row in range(feature_size):
        for col in range(feature_size):
            cx = (col + 0.5) * stride
            cy = (row + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * sqrt(ratio)
                    h = scale / sqrt(ratio)
                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2

                    anchors.append([x_min, y_min, x_max, y_max])

    return anchors
                    