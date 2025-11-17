from math import hypot

import cv2
import numpy as np

input_path = "input.jpg"
output_overlay = "detected_overlay.jpg"
output_rectified = "rectified_A4.jpg"

A4_ratio = 297 / 210  # 长宽比


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, ratio=A4_ratio):
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


def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    A = np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]])
    B = np.array([x3 - x1, y3 - y1])
    try:
        t = np.linalg.solve(A, B)
        xi = x1 + t[0] * (x2 - x1)
        yi = y1 + t[0] * (y2 - y1)
        return [xi, yi]
    except:
        return [0, 0]


def extreme_lines_ransac(lines, axis=1, top=True):
    """
    axis=1 -> y轴 (水平线)
    axis=0 -> x轴 (垂直线)
    top=True -> 选择最小位置（上或左），False -> 最大位置（下或右）
    """
    if len(lines) == 0:
        return None
    lines = np.array(lines)
    pos = np.min(lines[:, [axis, axis + 2]], axis=1) if top else np.max(lines[:, [axis, axis + 2]], axis=1)
    idx = np.argmin(pos) if top else np.argmax(pos)
    return lines[idx]

# 这里使用imread函数读取图像，要注意路径不能有中文，否则会报错
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"无法读取 {input_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
if lines is None:
    raise RuntimeError("HoughLinesP 没有检测到任何线段")

lines = lines[:, 0, :]  # reshape

horizontals, verticals = [], []
for x1, y1, x2, y2 in lines:
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    if abs(angle) < 20:
        horizontals.append([x1, y1, x2, y2])
    elif abs(angle - 90) < 20 or abs(angle + 90) < 20:
        verticals.append([x1, y1, x2, y2])

if len(horizontals) < 2 or len(verticals) < 2:
    raise RuntimeError("未检测到足够的水平或垂直线，请调整拍摄角度或参数")

h_top = extreme_lines_ransac(horizontals, axis=1, top=True)
h_bottom = extreme_lines_ransac(horizontals, axis=1, top=False)
v_left = extreme_lines_ransac(verticals, axis=0, top=True)
v_right = extreme_lines_ransac(verticals, axis=0, top=False)

pts = np.array(
    [
        line_intersection(h_top, v_left),
        line_intersection(h_top, v_right),
        line_intersection(h_bottom, v_right),
        line_intersection(h_bottom, v_left),
    ],
    dtype="float32",
)

overlay = img.copy()
for pt in pts:
    cv2.circle(overlay, tuple(np.int32(pt)), 6, (0, 0, 255), -1)
for i in range(4):
    cv2.line(overlay, tuple(np.int32(pts[i])), tuple(np.int32(pts[(i + 1) % 4])), (0, 255, 0), 2)
cv2.imwrite(output_overlay, overlay)

warped = four_point_transform(img, pts)
cv2.imwrite(output_rectified, warped)

print(f"检测结果已保存：{output_overlay}")
print(f"透视校正结果已保存：{output_rectified}")
