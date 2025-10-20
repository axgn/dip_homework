from math import hypot

import cv2
import numpy as np

input_path = "input.jpg"
output_overlay = "detected_overlay.jpg"
output_rectified = "rectified_A4.jpg"


def order_points(pts):
    """按 [左上, 右上, 右下, 左下] 顺序排列四个角点"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, ratio=297 / 210):  # A4长宽比≈1.414
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = int(max(widthA, widthB))

    heightA = hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = int(max(heightA, heightB))

    if maxWidth > maxHeight:
        target_w, target_h = maxWidth, int(maxWidth / ratio)
    else:
        target_h, target_w = maxHeight, int(maxHeight / ratio)

    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_w, target_h))
    return warped


# 这里使用imread函数读取图像，要注意路径不能有中文，否则会报错
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"无法读取 {input_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 50, 150)
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

quad = None
for cnt in contours[:10]:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) == 4 and cv2.contourArea(approx) > 1000:
        quad = approx.reshape(4, 2)
        break

if quad is None:
    rect = cv2.minAreaRect(contours[0])
    quad = cv2.boxPoints(rect)
    quad = np.int32(quad)

overlay = image.copy()
cv2.drawContours(overlay, [np.int32(quad)], -1, (0, 255, 0), 3)
for x, y in np.int32(quad):
    cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)

cv2.imwrite(output_overlay, overlay)

warped = four_point_transform(image, quad)
cv2.imwrite(output_rectified, warped)

print(f"检测结果已保存：{output_overlay}")
print(f"透视校正结果已保存：{output_rectified}")
