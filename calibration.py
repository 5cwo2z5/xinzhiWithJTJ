import cv2
import sys
import copy
import math
import numpy as np
from utils import pos2box, read_boxes_rgb

def rgb2hsi(rgb_lst):

    #Calculate Intensity
    def calc_intensity(red, green, blue):
        return np.divide(blue + green + red, 3)

    #Calculate Saturation
    def calc_saturation(red, green, blue):
        minimum = np.minimum(np.minimum(red, green), blue)
        saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

        return saturation

    #Calculate Hue
    def calc_hue(red, green, blue):
        hue = np.copy(red)

        hue = 0.5 * ((red - green) + (red - blue)) / \
                    math.sqrt((red - green)**2 + ((red - blue) * (green - blue)))
        hue = math.acos(hue)

        if blue <= green:
            hue = hue
        else:
            hue = ((360 * math.pi) / 180.0) - hue

        return hue
    
    hsi_lst = []
    for (r, g, b) in rgb_lst:
        h = calc_hue(r, g, b)
        s = calc_saturation(r, g, b)
        i = calc_intensity(r, g, b)
        hsi_lst.append([h, s, i])
    return np.array(hsi_lst)
    
def plot_rect(img, x, y, SQUARE_SIZE):
    x1, y1 = x, y
    x2, y2 = x + SQUARE_SIZE, y + SQUARE_SIZE
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 1)

def mian():
    # Read data
    img = cv2.imread(sys.argv[1])
    img_dbg = copy.deepcopy(img)

    # Initialize constants
    ROWS, COLS, DIMS = img.shape
    SQUARE_SIZE = 100
    MARGIN = 100
    COL_RANGE = COLS - MARGIN * 2 + 1
    ROW_RANGE = ROWS - MARGIN * 2 + 1

    # Calculate square position list and plot
    pos_lst = []
    for y in range(MARGIN, ROW_RANGE, SQUARE_SIZE):
        row_lst = []
        for x in range(MARGIN, COL_RANGE, SQUARE_SIZE):
            row_lst.append([x, y])
            plot_rect(img_dbg, x, y, SQUARE_SIZE)
        pos_lst.append(row_lst)
    pos_array = np.array(pos_lst)

    # Calculate hsi calib array
    box_lst = pos2box(pos_array.reshape(-1, pos_array.shape[-1]), SQUARE_SIZE, SQUARE_SIZE)
    rgb_lst = read_boxes_rgb(img, box_lst, [1, 1, 1])
    rgb_array = np.array(rgb_lst)
    r = rgb_array[:, 0].reshape(pos_array.shape[:2])
    g = rgb_array[:, 1].reshape(pos_array.shape[:2])
    b = rgb_array[:, 2].reshape(pos_array.shape[:2])
    r_aver = np.average(r)
    g_aver = np.average(g)
    b_aver = np.average(b)
    r_calib_array = r / r_aver
    g_calib_array = g / g_aver
    b_calib_array = b / b_aver

    # Print calibration data to screen
    # print(r_calib_array.tolist(), ';')
    # print(g_calib_array.tolist(), ';')
    # print(b_calib_array.tolist())

    # Put calib data in image for debug
    for calib, (x, y) in zip (g_calib_array.flatten(), pos_array.reshape(-1, pos_array.shape[-1])):
        cv2.putText(img_dbg, "%.2f" % calib, (x, y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    cv2.imwrite('./python/xinzhi/coeff_calib.jpg', img_dbg)
    # Write data to file
    # of = open("rgb.txt", "w")
    # for (r, g, b) in rgb_array:
    #     of.write("%.2f\t%.2f\t%.2f\n" % (r, g, b))
    # of.close()

    # of = open("calib_array.txt", "w")
    # of.write(str(r_calib_array.tolist()) + '\n')
    # of.write(str(g_calib_array.tolist()) + '\n')
    # of.write(str(b_calib_array.tolist()) + '\n')
    # of.close()

    # Test calibration coefficients
    sample_boxes = [
        [290, 660, 20, 20],
        [700, 1395, 20, 20],
        [397, 896, 20, 20]
    ]
    rgb_lst = read_boxes_rgb(img, sample_boxes, [1, 1, 1])
    print("raw ", rgb_lst)
    for sample_box, (r, g, b) in zip(sample_boxes, rgb_lst):
        plot_rect(img_dbg, sample_box[0], sample_box[1], sample_box[2])
        uleft_pt = [sample_box[0], sample_box[1]]
        uright_pt = [sample_box[0] + sample_box[2], sample_box[1]]
        lleft_pt = [sample_box[0], sample_box[1] + sample_box[3]]
        lright_pt = [sample_box[0] + sample_box[2], sample_box[1] + sample_box[3]]

        for (x, y) in pos_array.reshape(-1, pos_array.shape[-1]):
            if 0 <= x - uleft_pt[0] < SQUARE_SIZE and 0 <= y - uleft_pt[1] < SQUARE_SIZE:
                cv2.circle(img_dbg, (x, y), 5, (255, 0, 0), -1)

                uleft_diff_x  = x - uleft_pt[0]
                uleft_diff_y  = y - uleft_pt[1]
                uright_diff_x = x - uright_pt[0]
                uright_diff_y = y - uright_pt[1]
                lleft_diff_x  = x - lleft_pt[0]
                lleft_diff_y  = y - lleft_pt[1]
                lright_diff_x = x - lright_pt[0]
                lright_diff_y = y - lright_pt[1]
 
                area_lst = [
                    (20+uright_diff_x if uright_diff_x<0 else 20) * (20+lleft_diff_y if lleft_diff_y<0 else 20), 
                    (-uright_diff_x if uright_diff_x<0 else 0) * (20+lleft_diff_y if lleft_diff_y<0 else 20), 
                    (20+lright_diff_x if lright_diff_x<0 else 20) * (-lleft_diff_y if lleft_diff_y<0 else 0), 
                    (-lright_diff_x if lright_diff_x<0 else 0) * (-lright_diff_y if lright_diff_y<0 else 0)
                ] 
                print(x, y)
                print(area_lst)
                uleft_weight  = area_lst[0] / 20 ** 2
                uright_weight = area_lst[1] / 20 ** 2
                lleft_weight  = area_lst[2] / 20 ** 2
                lright_weight = area_lst[3] / 20 ** 2
                
                cpt_index = pos_array.reshape(-1, pos_array.shape[-1]).tolist().index([x, y])
                cpt_row = cpt_index // pos_array.shape[0]
                cpt_col = cpt_index % pos_array.shape[0]

                r_coeff_calib = r_calib_array[cpt_row-1][cpt_col-1] * uleft_weight \
                            + r_calib_array[cpt_row-1][cpt_col] * uright_weight \
                            + r_calib_array[cpt_row][cpt_col-1] * lleft_weight \
                            + r_calib_array[cpt_row][cpt_col] * lright_weight
                g_coeff_calib = g_calib_array[cpt_row-1][cpt_col-1] * uleft_weight \
                            + g_calib_array[cpt_row-1][cpt_col] * uright_weight \
                            + g_calib_array[cpt_row][cpt_col-1] * lleft_weight \
                            + g_calib_array[cpt_row][cpt_col] * lright_weight
                b_coeff_calib = b_calib_array[cpt_row-1][cpt_col-1] * uleft_weight \
                            + b_calib_array[cpt_row-1][cpt_col] * uright_weight \
                            + b_calib_array[cpt_row][cpt_col-1] * lleft_weight \
                            + b_calib_array[cpt_row][cpt_col] * lright_weight

                print("校准系数", [r_coeff_calib, g_coeff_calib, b_coeff_calib])
                print("校准后的值", [1/r_coeff_calib*r, 1/g_coeff_calib*g, 1/b_coeff_calib*b])




if __name__ == "__main__":
    mian()
