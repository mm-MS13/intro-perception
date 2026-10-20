import sys
import cv2 as cv
import numpy as np

path = sys.argv[1]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

img = cv.imread(path)
if img is None:
    raise SystemExit(f"could not read {path}")
img = cv.resize(img, None, fx=scale, fy=scale)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

WIN = "viewer"
cv.namedWindow(WIN)
sliders = [("H min", 0, 180), ("H max", 10, 180),
           ("S min", 100, 255), ("S max", 255, 255),
           ("V min", 100, 255), ("V max", 255, 255)]
for name, start, top in sliders:
    cv.createTrackbar(name, WIN, start, top, lambda v: None)

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and x < img.shape[1]:
        print(f"({x},{y}) HSV = {hsv[y, x]}")

cv.setMouseCallback(WIN, on_mouse)

while True:
    pos = {n: cv.getTrackbarPos(n, WIN) for n, _, _ in sliders}
    lo = np.array([pos["H min"], pos["S min"], pos["V min"]])
    hi = np.array([pos["H max"], pos["S max"], pos["V max"]])
    mask = cv.inRange(hsv, lo, hi)
    both = np.hstack((img, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)))
    cv.imshow(WIN, both)
    if cv.waitKey(30) in (ord("q"), 27):
        break
cv.destroyAllWindows()