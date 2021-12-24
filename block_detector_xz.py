from __future__ import print_function

import sys
import cv2
import math
import numpy as np
from PIL import Image, ImageDraw
from pyzbar.pyzbar import decode, ZBarSymbol

from block_position import relative_position_photo

def get_threshold(img):
  thres = 100
  img = cv2.medianBlur(img, 5)
  ret, th = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
  return th

def is_black_or_white(th, x, y):
# 0 refers black, 255 refers white
  if th[x][y] == 0:
    return 0
  else:
    return 1

def is_all_white(th, x=-1, y=0, square_size=0):
  if x == -1:
    return (th == 255).all()
  else:
    sqa = th[x:x+square_size, y:y+square_size]
    return (sqa == 255).all()

def has_black_px(sqa, min_amount_of_black_px):
  amount_of_black_px = np.count_nonzero(sqa==0)
  if amount_of_black_px > min_amount_of_black_px:
    return True
  else:
    return False

def get_radian(a, b, c):
  rad = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
  return -rad

def central(points):
  sumx = sumy = 0
  for x, y in points:
    sumx += x
    sumy += y
  return (sumx/len(points), sumy/len(points))

def distance(p1, p2):
  x1, y1 = p1
  x2, y2 = p2
  return math.hypot(x2-x1, y2-y1)

def get_qr_data(img):
  qr_data = decode(img, symbols=[ZBarSymbol.QRCODE])
  return qr_data

def custom_blur_demo(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
  img = cv2.filter2D(image, -1, kernel=kernel)
  #cv2.imshow("custom_blur_demo", img)
  #cv2.waitKey()
  return img

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
  """Return a sharpened version of the image, using an unsharp mask."""
  blurred = cv2.GaussianBlur(image, kernel_size, sigma)
  sharpened = float(amount + 1) * image - float(amount) * blurred
  sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
  sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
  sharpened = sharpened.round().astype(np.uint8)
  if threshold > 0:
    low_contrast_mask = np.absolute(image - blurred) < threshold
    np.copyto(sharpened, image, where=low_contrast_mask)
  return sharpened
  
def bounding_box_and_polygon(img_path):
  image = Image.open(img_path).convert('RGB')
  draw = ImageDraw.Draw(image)
  for barcode in get_qr_data(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)):
    rect = barcode.rect
    draw.rectangle(
        (
            (rect.left, rect.top),
            (rect.left + rect.width, rect.top + rect .height)
        ),
        outline='#0080ff'
    )
    draw.polygon(barcode.polygon, outline='red')
  image.save('bounding_box_and_polygon.png')

def get4point(qr_data):
  pos4 = {}
  for qd in qr_data:
    pos4[qd.data] = central(qd.polygon)
  return pos4

def update_scale(pos4, DIST):
  point1 = pos4[b'2']
  point2 = pos4[b'4']
  dist_x = distance(point1, point2)
  print(dist_x)
  # input()
  # scale_y = dist_y / DIST[0]
  # scale_x = dist_x / DIST[1]

  # return (scale_y, scale_x)

def update_radian(point_mid, point_src):
  x3, y3 = point_mid
  x4, y4 = point_src
  rad = get_radian(point_src, point_mid, (x3+1, y3))
  return rad

def get_point_by_radian(dist, radian, point0=(0, 0)):
  x0, y0 = point0
  X = dist * math.cos(radian) + x0
  Y = dist * math.sin(radian) + y0
  return (X, Y)

def qr_detector(img, STEP_SIZE, SQUARE_SIZE):
  row, col, dim = img.shape
  x = y = 0
  posdict = {}
  while True:
    pos = [x, y]
    if y + SQUARE_SIZE > row:
      break
    crop_img = img[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
    qrd = get_qr_data(crop_img)
    if qrd:
      cx, cy = central(qrd[0].polygon)
      posdict[qrd[0].data] = (round(cx+x), round(cy+y))
    if x + SQUARE_SIZE <= col:
      x = x + STEP_SIZE
    else:
      y = y + STEP_SIZE
      x = 0
  return posdict

def draw_point(img_path, relative_position_1, SCALE, RADIAN, src_point):
  img = cv2.imread(img_path)
  for block, rd_value in relative_position_1.items():
    point = get_point_by_radian(rd_value['dist']*SCALE[1], rd_value['rad']+RADIAN, src_point)
    point = tuple([round(x) for x in point])
    print(point)
    cv2.circle(img, point, 2, (0, 0, 255), 0)
  cv2.imwrite('test.jpg', img)

def main():

  '''yaxis = 1947
  points = (
    (839, yaxis),
    (956, yaxis),
    (1073, yaxis),
    (1190, yaxis),
    (1307, yaxis),
    (1425, yaxis),
    (1541, yaxis),
  )
  p1 = (648.5, 713.5)
  i = 1
  for p in points:
    RADIAN = update_radian(p1, p)
    dist = distance(p, p1)
    print("'b43%d' : {'rad' : %.3f, 'dist': %.3f}," % (i, RADIAN, dist))
    i += 1
  input()'''

  # data initialize
  DIST = [DIST_Y, DIST_X] = (570, 618)
  SCALE = [SCALE_Y, SCALE_X] = (1, 1)
  RADIAN = 0
  STEP_SIZE = 30
  SQUARE_SIZE = 180

  img_path = sys.argv[1]
  img = cv2.imread(img_path)
  # img = unsharp_mask(img)
  posn = qr_detector(img, STEP_SIZE, SQUARE_SIZE)
  print(posn)
  print(decode(img, symbols=[ZBarSymbol.QRCODE]))

  if len(posn) == 4:
    SCALE = update_scale(posn, DIST)
    RADIAN = update_radian(posn[b'1'], posn[b'2'])
    print(RADIAN)
    draw_point(img_path, relative_position_photo, SCALE, RADIAN, posn[b'1'])

if __name__ == "__main__":
  main()
