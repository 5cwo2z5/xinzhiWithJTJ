import sys
import cv2
import math
import json
import numpy as np
from block_detector_xz import update_radian, get_point_by_radian
from block_position import relative_position_photo
from color_data import *
from utils import *


def most_approx(src_color, dst_color):

    def distance(c1, c2):
        (r1, g1, b1) = c1
        (r2, g2, b2) = c2
        return math.sqrt((r1 - r2)**2 + (g1 - g2) ** 2 + (b1 - b2) **2)

    approx = [distance(src_color, dst) for dst in dst_color]
    return approx.index(min(approx))


def xinzhi_process(posn):

    def plot_rect(img, x, y, SQUARE_SIZE):
        x1, y1 = x, y
        x2, y2 = x + SQUARE_SIZE, y + SQUARE_SIZE
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def plot_rects(img, box_list):
        for box in box_list:
            x1, y1 = box[0], box[1]
            x2, y2 = box[0]+box[2], box[1]+box[3]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    def get_position(img_path, relative_position_1, SCALE, RADIAN, src_point):
        points = []
        for _, rd_value in relative_position_1.items():
            point = get_point_by_radian(rd_value['dist']*SCALE, rd_value['rad']+RADIAN, src_point)
            point = list([int(round(x)) for x in point])
            points.append(point)
        return points

    def get_color_shift(std_color, calib_rgb):
        tgt_color = [
            (calib_rgb[0][0]+calib_rgb[1][0])//2,
            (calib_rgb[0][1]+calib_rgb[1][1])//2,
            (calib_rgb[0][2]+calib_rgb[1][2])//2
        ]
        rshift = std_color[0] / tgt_color[0]
        gshift = std_color[1] / tgt_color[1]
        bshift = std_color[2] / tgt_color[2]
        return [rshift, gshift, bshift]

    def calc_color_calib_coeff(a_calib_color, cpt_index_2d, l_weight):
        return a_calib_color[cpt_index_2d[0]-1][cpt_index_2d[1]-1]*l_weight[0] \
                + a_calib_color[cpt_index_2d[0]-1][cpt_index_2d[1]]*l_weight[1] \
                + a_calib_color[cpt_index_2d[0]][cpt_index_2d[1]-1]*l_weight[2] \
                + a_calib_color[cpt_index_2d[0]][cpt_index_2d[1]]*l_weight[3]

    def calc_calibration_rgb_array_list():
        s_calib = sys.argv[2:]
        l_calib_red = list(map(lambda x: x.replace('[', '').replace(']', '').strip(), s_calib[0].split(',')))
        l_calib_grn = list(map(lambda x: x.replace('[', '').replace(']', '').strip(), s_calib[1].split(',')))
        l_calib_blu = list(map(lambda x: x.replace('[', '').replace(']', '').strip(), s_calib[2].split(',')))
        a_calib_red = np.array(l_calib_red, dtype=np.double) \
            .reshape(len(s_calib[0].split('], [')), len(s_calib[0].split('], [')[0].split(',')))
        a_calib_grn = np.array(l_calib_grn, dtype=np.double) \
            .reshape(len(s_calib[1].split('], [')), len(s_calib[1].split('], [')[0].split(',')))
        a_calib_blu = np.array(l_calib_blu, dtype=np.double) \
            .reshape(len(s_calib[2].split('], [')), len(s_calib[2].split('], [')[0].split(',')))
        return [a_calib_red, a_calib_grn, a_calib_blu]

    def rgb_calibration(img_dbg, l_box, l_rgb, a_calib_rgb):

        # Calculate square position list
        l_calib_pos = []
        for y in range(MARGIN, ROW_RANGE, CALIB_SQUARE_SIZE):
            l_row = []
            for x in range(MARGIN, COL_RANGE, CALIB_SQUARE_SIZE):
                plot_rect(img_dbg, x, y, CALIB_SQUARE_SIZE)
                l_row.append([x, y])
            l_calib_pos.append(l_row)
        a_calib_pos = np.array(l_calib_pos)

        l_rgb_calibed = []
        for box, (r, g, b) in zip(l_color_box, l_rgb):
            pt_uleft = [box[0], box[1]]
            pt_uright = [box[0] + box[2], box[1]]
            pt_lleft = [box[0], box[1] + box[3]]
            pt_lright = [box[0] + box[2], box[1] + box[3]]

            for (x, y) in a_calib_pos.reshape(-1, a_calib_pos.shape[-1]):
                if 0 <= x - pt_uleft[0] < CALIB_SQUARE_SIZE and 0 <= y - pt_uleft[1] < CALIB_SQUARE_SIZE:

                    diff_x_uright = x - pt_uright[0]
                    diff_y_lleft  = y - pt_lleft[1]
                    diff_x_lright = x - pt_lright[0]
                    diff_y_lright = y - pt_lright[1]

                    l_area = [
                        (XZ_SQUARE_SIZE+diff_x_uright if diff_x_uright<0 else XZ_SQUARE_SIZE) \
                            *(XZ_SQUARE_SIZE+diff_y_lleft if diff_y_lleft<0 else XZ_SQUARE_SIZE),
                        (-diff_x_uright if diff_x_uright<0 else 0) \
                            *(XZ_SQUARE_SIZE+diff_y_lleft if diff_y_lleft<0 else XZ_SQUARE_SIZE),
                        (XZ_SQUARE_SIZE+diff_x_lright if diff_x_lright<0 else XZ_SQUARE_SIZE) \
                            *(-diff_y_lleft if diff_y_lleft<0 else 0),
                        (-diff_x_lright if diff_x_lright<0 else 0) \
                            *(-diff_y_lright if diff_y_lright<0 else 0)
                    ]

                    weight_uleft  = l_area[0] / XZ_SQUARE_SIZE ** 2
                    weight_uright = l_area[1] / XZ_SQUARE_SIZE ** 2
                    weight_lleft  = l_area[2] / XZ_SQUARE_SIZE ** 2
                    weight_lright = l_area[3] / XZ_SQUARE_SIZE ** 2
                    l_weight = [weight_uleft, weight_uright, weight_lleft, weight_lright]

                    cpt_index = a_calib_pos.reshape(-1, a_calib_pos.shape[-1]).tolist().index([x, y])
                    cpt_row = cpt_index // a_calib_pos.shape[0]
                    cpt_col = cpt_index % a_calib_pos.shape[0]
                    cpt_index_2d = [cpt_row, cpt_col]

                    coeff_calib_red = calc_color_calib_coeff(a_calib_rgb[0], cpt_index_2d, l_weight)
                    coeff_calib_grn = calc_color_calib_coeff(a_calib_rgb[1], cpt_index_2d, l_weight)
                    coeff_calib_blu = calc_color_calib_coeff(a_calib_rgb[2], cpt_index_2d, l_weight)

                    l_rgb_calibed.append([1/coeff_calib_red*r, 1/coeff_calib_grn*g, 1/coeff_calib_blu*b])

        return l_rgb_calibed


    # Read image
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    img_dbg = cv2.imread(img_path)

    # Initialize constants
    DEBUG_MODE = 0

    SCALE = 1
    RADIAN = 0
    XZ_SQUARE_SIZE = 20
    STD_COLOR = [146, 146, 133]
    ROWS, COLS, DIMS = img.shape
    CALIB_SQUARE_SIZE = 100
    MARGIN = 100
    COL_RANGE = COLS - MARGIN * 2 + 1
    ROW_RANGE = ROWS - MARGIN * 2 + 1
    RADIAN = update_radian(posn[b'2'], posn[b'4'])

    
    # cv2.circle(img_dbg, posn[b'2'], 5, (255, 0, 0), -1)
    # cv2.circle(img_dbg, posn[b'4'], 5, (255, 0, 0), -1)
    # cv2.circle(img_dbg, (int((posn[b'4'][0]+posn[b'2'][0])/2), int((posn[b'4'][1]+posn[b'2'][1])/2)), 5, (255, 0, 0), -1)
    # cv2.line(img_dbg, , (255, 0, 0))
    l_pos = get_position(img_path, relative_position_photo, SCALE, RADIAN, posn[b'2'])
    if len(l_pos) == 17:

        # l_a_calib_rgb = calc_calibration_rgb_array_list()

        l_color_box = pos2box(l_pos[1:15], XZ_SQUARE_SIZE, XZ_SQUARE_SIZE)
        l_calib_box = pos2box(l_pos[15:], XZ_SQUARE_SIZE, XZ_SQUARE_SIZE)
        l_calib_rgb_src = read_boxes_rgb(img, l_calib_box, [1, 1, 1])
        l_calib_rgb_dst = l_calib_rgb_src#rgb_calibration(img_dbg, l_calib_box, l_calib_rgb_src, l_a_calib_rgb)
        color_shift = get_color_shift(STD_COLOR, l_calib_rgb_dst)
        l_rgb = read_boxes_rgb(img, l_color_box, color_shift)
        plot_rects(img_dbg, np.concatenate((l_color_box, l_calib_box), axis=0))

        l_rgb_calibed = l_rgb#rgb_calibration(img_dbg, l_color_box, l_rgb, l_a_calib_rgb)
        cv2.imwrite('./python/xinzhi/coeff_calib.jpg', img_dbg)

        if DEBUG_MODE:
            for rgb, (key, value) in zip(l_rgb_calibed, standrad_data.items()):
                print(key, rgb)
            print(l_calib_rgb_dst)
            exit(0)

        if l_rgb:
            result_dict = {}
            result_str = ''
            for src, (key, value) in zip(l_rgb_calibed, standrad_data.items()):
                result_dict[key] = rst_lst[key][most_approx(src, value)]
            for (k, v) in result_dict.items():
                result_str += str(k) + ':' + str(v) + '\n'
                jsondict = json.dumps(result_dict)
            print(jsondict)
            return

    print('error')
