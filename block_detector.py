import sys
import cv2
import numpy as np
from src_data import boxesT as boxes

def get_threshold(img):
  thres = 20
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

def is_black_beside(th, pos, square_size, min_amount_of_black_px):
  udlr = {}
  x, y = pos
  u = th[x-square_size:x, y:y +square_size]
  d = th[x+square_size:x+2*square_size, y:y+square_size]
  l = th[x:x+square_size, y-square_size:y]
  r = th[x:x+square_size, y+square_size:y+2*square_size]
  for n, a in zip(('u', 'd', 'l', 'r'),(u, d, l, r)):
    udlr[n] = has_black_px(a, min_amount_of_black_px)
  if list(udlr.values()).count(True) == 4:
    return True
  else:
    return False

def detect_processor(th, step_size, square_size, min_amount_of_black_px):
  row, col = th.shape
  # print(row, col)
  x = y = square_size
  around_tect_sig = False
  times_no_black = 0
  detect_block_pos = []
  while True:
    pos = [x, y]
    if y + step_size > row:
      break
    sqa = th[x:x+square_size,y:y+square_size]
    if has_black_px(sqa, min_amount_of_black_px):
      around_tect_sig = True
      times_no_black = 0
    if times_no_black > 20 and around_tect_sig == True:
      around_tect_sig = False
    if around_tect_sig == True and is_all_white(sqa):
      if is_black_beside(th, pos, square_size, min_amount_of_black_px):
        detect_block_pos.append(pos)
    if x + step_size <= col:
      x = x + step_size
    else:
      y = y + step_size
      x = square_size
    times_no_black += 1
  return detect_block_pos

def merge_near_rect(origin_list, square_size, outcome=[]):
  group_pos = outcome
  group_list = [origin_list[0]]
  for pos in origin_list[1:]:
    y, x = pos
    for grouped_pos in group_list:
      Y, X = grouped_pos
      if (abs(x - X) <= square_size and abs(y - Y) <= square_size
            and pos not in group_list):
        group_list.append(pos)
  group_pos.append(list(np.mean(np.array(group_list), axis=0, dtype=np.integer)))
  rest_list = [i for i in origin_list if i not in group_list]
  if rest_list:
    merge_near_rect(rest_list, square_size, group_pos)
  return group_pos

def draw_rect(img, detect_block_pos, square_size, margin, file_name):
  for pos in detect_block_pos:
    x, y = pos[0], pos[1]
    #x += margin
    start_point = (x, y)
    end_point = (x+30, y+20)
    color = (0, 0, 255)
    thickness = 1
    img = cv2.rectangle(img, start_point, end_point, color, thickness) 
  cv2.imwrite(file_name, img)
  #cv2.imshow('t', img)
  #cv2.waitKey()

def crop_image(img, margin):
  return img[:, margin:-margin]

def position_in_origin_img(pos_list, margin, square_size):
  real_pos_list = []
  for pos in pos_list:
    pos[1] += margin
    pos[0], pos[1] = pos[1], pos[0]
    pos.extend([square_size, square_size])
    real_pos_list.append(pos)
  return real_pos_list

def pos_boxes(pos_list, boxes):
  head = min(y[1] for y in pos_list)
  e = [x for x in pos_list if head in x][0]
  X = pos_list[pos_list.index(e)][0] - boxes[0][0]
  Y = head - boxes[0][1]
  new_boxes = []
  for box in boxes:
    box[0] += X
    box[1] += Y
    new_boxes.append(box)
  return new_boxes

def calculate_boxes(pos_list, square_size):
  sorted_list = sorted(pos_list, key=lambda x: x[1])
  x_shift = sorted_list[1][0] - sorted_list[0][0]
  y_shift = sorted_list[1][1] - sorted_list[0][1]
  new_boxes = [sorted_list[0], sorted_list[1]]
  for i in range(12):
    i += 1
    new_boxes.append([new_boxes[i][0]+x_shift, new_boxes[i][1]+y_shift, square_size, square_size-10])
  return new_boxes

def main():
  step_size = 5
  square_size = 30 # 80px for phone, 15px for 1080p
  min_amount_of_black_px = 60 # just test
  margin = 10

  img_path = sys.argv[1]
  img = cv2.imread(img_path, 0)
  img = crop_image(img, margin)
  th = get_threshold(img)
  detect_block_pos = detect_processor(th, step_size, square_size, min_amount_of_black_px)
  # print(detect_block_pos)
  # of = open('block_pos.txt', 'w')
  if detect_block_pos:
    positions = merge_near_rect(detect_block_pos, square_size)
    positions = position_in_origin_img(positions, margin, square_size)
    # positions = pos_boxes(positions, boxes)
    positions = calculate_boxes(positions, square_size)
    print(positions)
    # if len(positions) == 14:
    #   of.write(str(positions))
    # else:
    #   of.write("0")
    # of.close()
    img = cv2.imread(img_path)
    draw_rect(img, positions, square_size, margin, 'position.jpg')
  # else:
    # of.write("0")

if __name__ == "__main__":
  main()
