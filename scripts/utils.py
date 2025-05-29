import cv2

def draw_boxes(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
    img = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if labels:
            cv2.putText(img, labels[i], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img