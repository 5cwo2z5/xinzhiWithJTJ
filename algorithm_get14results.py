# -*- coding: utf-8 -*-

import sys
import ast
import numpy as np
import json
import cv2
from PIL import Image
from src_data import boxesT as boxes
from block_detector import crop_image, get_threshold, detect_processor, merge_near_rect
from block_detector import position_in_origin_img, pos_boxes, draw_rect

URO = (
    (89, 102, 85),
    (109, 90, 44),
    (104, 77, 27),
    (106, 73, 18),
    (101, 68, 18)
)
BLO = (
    (117, 131, 105),
    (90, 110, 80),
    (30, 85, 70),
    (10, 60, 60)
)
BIL = (
    (102, 112, 85),
    (107, 99, 80),
    (110, 90, 75),
    (109, 64, 59)
)
KET = (
    (102, 113, 90),
    (87, 83, 88),
    (75, 59, 79),
    (59, 39, 73)
)
CA  = (
    (114, 110, 27),
    (109, 102, 24),
    (95, 71, 44),
    (92, 66, 53)
)
LEU = (
    (112, 124, 101),
    (107, 112, 98),
    (89, 87, 89),
    (76, 74, 84),
    (52, 39, 57)
)


GLU = (
    (136, 75, 98),
    (106, 62, 87),
    (74, 49, 84),
    (32, 33, 68),
    (20, 27, 53),
    (16, 29, 55)
)
PRO = (
    (129, 127, 52),
    (118, 126, 55),
    (112, 117, 58),
    (91, 114, 59),
    (69, 108, 75),
    (60, 103, 79)
)
PRO_BG = (
  (139, 159, 150),
  (135, 160, 156),
  (140, 160, 151),
  (139, 160, 151),
  (130, 161, 156),
  (138, 159, 151)
)
PH  = (
    (127, 83, 51),
    (132, 101, 44),
    (80, 99, 40),
    (34, 81, 53),
    (19, 77, 62)
)
CRE = (
    (115, 121, 51),
    (86, 115, 58),
    (49, 102, 60),
    (48, 97, 63),
    (46, 91, 67)
)
NIT = (
    (115, 131, 117),
    (113, 121, 113),
    (108, 106, 113)
)
SG  = (
    (19, 56, 76),
    (40, 66, 46),
    (52, 71, 38),
    (66, 81, 32),
    (86, 88, 27),
    (90, 90, 25),
    (85, 91, 32)
)
VC  = (
    (15, 75, 80),
    (17, 78, 63),
    (73, 104, 60),
    (107, 120, 60),
    (118, 124, 65)
)
MCA = (
    (105, 121, 109),
    (102, 123, 115),
    (94, 117, 106),
    (93, 119, 108)
)
standrad_data = {
    'URO': URO,
    'BLO': BLO,
    'BIL': BIL,
    'KET': KET,
    'CA': CA,
    'LEU': LEU,
    'GLU': GLU,
    'PRO': PRO,
    'PH' : PH,
    'CRE': CRE,
    'NIT': NIT,
    'SG': SG,
    'VC': VC,
    'MCA': MCA
}
data0 = {
    "URO": "",
    "BLO": "-",
    "BIL": "",
    "KET": "",
    "CA": "",
    "LEU": "",
    "GLU": "",
    "PRO": "",
    "PH": "",
    "CRE": "",
    "NIT": "",
    "SG": "",
    "VC": "",
    "MCA": ""
}

rst_lst = {
    'URO': ('-', '+', '++', '+++', '++++'),
    'BLO': ('-', '+', '++', '+++'),
    'BIL': ('-', '-', '+', '++'),
    'KET': ('-', '+', '++', '+++'),
    'CA': ('-', '-', '+', '++'),
    'LEU': ('-', '-', '微量', '少量', '中量'),#, '大量'),
    'GLU': ('-', '+-', '+', '++', '+++', '++++'),
    'PRO': ('-', '-', '+-', '+', '++', '+++'),#, '++++'),
    'PH': ('5', '6', '7', '8', '9'),
    'CRE': ('-', '-', '+-', '+', '++'),# '+++'),
    'NIT': ('-', '-', '弱阳性'),#强阳性
    'SG': ('1.000', '1.005', '1.010', '1.015', '1.020', '1.025', '1.030'),
    'VC': ('0', '0.6', '1.4', '2.8', '5.6'),
    'MCA': ('-', '+', '++', '+++')
}

def std(lst, x):
  """Calculate the standard deviation with the list and x.

  Parameters:
    lst: A list of tuple.
    x: A number given to be calculated the deviation with the list.

  Returns:
    A float number.
  """
  if len(lst) != 0:
    return np.sqrt(np.mean(abs(np.array(lst) - x)**2))
  else:
    print('The lenght of list is 0.')

def make_int_aver(lst):
  """Calculate the average of the list and round.
 
  Parameters:
    lst: A flat list that only contains number.

  Returns:
    An integer.
  """
  if len(lst) != 0:
    return round(sum(lst)/len(lst))
  else:
    print('The lenght of list is 0.')

def blk_lst_maker(rgb_arr):
  blk_lst = []
  for a in rgb_arr:
    if np.mean(a) < 55:
      blk_lst.append(1)
    else:
      blk_lst.append(0)
  return blk_lst

def color_approximation(src, dst):
  src, dst = map(np.array, (src, dst))
  x = np.dot(src, dst)/(np.sqrt(sum(src**2))*np.sqrt(sum(dst**2)))
  r = np.arccos(x)*(180/np.pi)
  return r

def most_approx(src_color, dst_color):
  approx = [color_approximation(src_color, dst) for dst in dst_color]
  return approx.index(min(approx))

def get_rgb_in_square(im, box):
  """Get rgb data from the square region.

  Parameters:
    im: An image object.
    box: A square region with (posx, posy, sizex, sizey).

  Returns:
    A list contain rgb data. For example:
    [ (157, 152, 156), (157, 152, 156), (157, 152, 156),
      (157, 152, 156), (157, 152, 156), (157, 152, 156) ]
  """
  posx, posy, sizex, sizey = box
  region = im.crop(box)
  lst = []
  for y in range(posy, posy+sizey):
    for x in range(posx, posx+sizex):
      r, g, b = im.getpixel((x, y))
      rgb = (r, g, b)
      lst.append(rgb)
  return lst

def fetch_aver_rgb(lst):
  """Calculate the average rgb of the rgb list.

  Parameters:
    lst: A list contain rgb data. For example:
    [ (157, 152, 156), (157, 152, 156) ]

  Returns:
    A tuple of average rgb.
  """
  if len(lst) != 0:
    r, g, b = [ [item[i] for item in lst] for i in range(len(lst[0])) ]
    return (make_int_aver(r), make_int_aver(g), make_int_aver(b))
  else:
    print('The lenght of list is 0.')

def get_main_color(im, box):
  """Get the main color of the region.

  Parameters:
    im: An image object.
    box: Four values of position and size.

  Returns:
    Main rgb value of the region.
  """
  color_lst = get_rgb_in_square(im, box)
  if color_lst == 0:
    print('No color data.')
    return 0
  r, g, b = [ [item[i] for item in color_lst] for i in range(len(color_lst[0])) ]
  
  rmax, gmax, bmax = map(max, (r, g, b))
  #print(rmax, gmax, bmax)
  std_rm = std(r, rmax)
  std_gm = std(g, gmax)
  std_bm = std(b, bmax)
  #print((std_rm, std_gm, std_bm))
  
  ravg, gavg, bavg = map(make_int_aver, (r, g, b))
  #print(ravg, gavg, bavg)
  std_ra = std(r, ravg)
  std_ga = std(g, gavg)
  std_ba = std(b, bavg)
  #print((std_ra, std_ga, std_ba))
  rmain = (rmax if std_ra > std_rm else ravg) #+ 40
  gmain = (gmax if std_ga > std_gm else gavg) #+ 30
  bmain = (bmax if std_ba > std_bm else bavg) #+ 30
  return (rmain if rmain <= 255 else 255, 
          gmain if gmain <= 255 else 255, 
          bmain if bmain <= 255 else 255)

def read_col_rgb(im, boxes):
  rgbs = []
  for box in boxes:
    c = get_main_color(im, box)
    #print(c)
    rgbs.append(c)
  return tuple(rgbs)

def get_boxes_from_str(boxes_str):
  boxes = ast.literal_eval(boxes_str)
  return boxes

def calculate_boxes(pos_list, square_size):
  sorted_list = sorted(pos_list, key=lambda x: x[1])
  y_coord = 7
  x_shift = sorted_list[1][0] - sorted_list[0][0]
  # print(x_shift)
  y_shift = sorted_list[1][1] - sorted_list[0][1]
  y_shift = int(y_shift * 0.98)
  x_shift = int(x_shift * 0.95)
  # print(x_shift, y_shift)
  new_boxes = [[sorted_list[0][0], sorted_list[0][1]+y_coord, square_size, square_size-10], 
               [sorted_list[1][0], sorted_list[1][1]+y_coord, square_size, square_size-10]]
  for i in range(12):
    i += 1
    new_boxes.append([new_boxes[i][0]+x_shift, new_boxes[i][1]+y_shift, square_size, square_size-10])
  return new_boxes

def strip_process():
  # get position
  step_size = 2
  square_size = 30
  min_amount_of_black_px = 60 # just test
  margin = 0

  img_path = sys.argv[1]
  # rotate 180
  # img = cv2.imread(img_path)
  # img = cv2.rotate(img, cv2.ROTATE_180)
  # cv2.imwrite(img_path, img)
  # -----------
  img = cv2.imread(img_path, 0)
  # img = crop_image(img, margin)
  th = get_threshold(img)
  detect_block_pos = detect_processor(
    th, step_size, square_size, min_amount_of_black_px)
  # print(detect_block_pos)
  if detect_block_pos:
    positions = merge_near_rect(detect_block_pos, square_size)
    positions = position_in_origin_img(positions, margin, square_size)
    positions = calculate_boxes(positions, square_size)
    # print(positions)
    if len(positions) == 14:
      # get resultss
      im = Image.open(img_path).convert('RGB')
      # boxesr = get_boxes_from_str(str(positions))
      rgb_datas = read_col_rgb(im, positions)
      if rgb_datas:
        r = {}
        rs = ''
        for src, (key, value) in zip(rgb_datas, standrad_data.items()):
          r[key] = rst_lst[key][most_approx(src, value)]
        for (k, v) in r.items():
          rs += str(k) + ':' + str(v) + '\n'
        jsondict = json.dumps(r)
        print(jsondict)
      else:
        print('no datas')
    else:
      print('error')
    img = cv2.imread(img_path)
    draw_rect(img, positions, square_size, margin, 'position.jpg')
  else:
    print('error')
  #print("参数1："+sys.argv[1])
  #print("参数2："+sys.argv[2])
  

if __name__ == "__main__":
  strip_process()
