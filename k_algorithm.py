# -*- coding:utf-8 -*-
"""
此文件是用来写单卡胶体金识别算法！
"""
import json
import scipy.signal
import numpy as np
import os
import cv2
import heapq
import math
import matplotlib.pyplot as plt
from kneed import KneeLocator

relative_position_photo_dkjtj = {
    'ja': {'rad': 0., 'dist': -355.000},
    'jb': {'rad': 0.06, 'dist': -355.000},
    'jc': {'rad': 0, 'dist': -175.000},
    'jd': {'rad': 0.09, 'dist': -175.000},
}


def getListMaxNumIndex(num_list, topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    # max_list = heapq.nlargest(topk, num_list)
    min_list = heapq.nsmallest(topk, num_list)
    # max_num_index = [num_list.index(max_list[i]) for i in range(topk)]
    min_num_index = [num_list.index(min_list[i]) for i in range(topk)]
    # print(max_list)
    # print('max_num_index:', max_num_index)
    a, b = min_num_index
    v1, v2 = min_list
    # print(min_list)
    # print('min_num_index:', min_num_index)
    return a, b, v1, v2


def dg_nal(y):
    if len(y) == 0:
        yy = [0, 0]
    else:
        yy = y
    return yy


#  接受到二维码中的定位进行找中点和定位
def get_point_by_radian(dist, radian, point0=(0, 0)):
    x0, y0 = point0
    X = dist * math.cos(radian) + x0
    Y = dist * math.sin(radian) + y0
    return (X, Y)


def get_position(img_path, relative_position_1, SCALE, RADIAN, src_point):
    points = []
    for _, rd_value in relative_position_1.items():
        point = get_point_by_radian(rd_value['dist'] * SCALE, rd_value['rad'] + RADIAN, src_point)  # 偏差归位
        point = list([int(round(x)) for x in point])
        points.append(point)
    return points


def get_radian(a, b, c):
    rad = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    return -rad


def updata_radian(point_mid, point_src):
    x3, y3 = point_mid
    x4, y4 = point_src
    rad = get_radian(point_src, point_mid, (x3+1, y3))
    return rad


def xinzhi_process_dkjtj(posn, img_path):
    SCALE = 1
    # img_dbg = cv2.imread(img_path)
    RADIAN = updata_radian(posn[b'2'], posn[b'5'])  # 计算与水平的偏度
    # cv2.circle(img_dbg, posn[b'2'], 5, (255, 0, 0), -1)
    # cv2.circle(img_dbg, posn[b'5'], 5, (255, 0, 0), -1)
    # cv2.circle(img_dbg, (int((posn[b'5'][0] + posn[b'2'][0]) / 2), int((posn[b'5'][1] + posn[b'2'][1]) / 2)), 5,
    #            (255, 0, 0), -1)
    l_pos = get_position(img_path, relative_position_photo_dkjtj, SCALE, RADIAN, posn[b'2'])  # 中心坐标

    # for iii in range(4):
    #     cv2.circle(img_dbg, (l_pos[iii][0], l_pos[iii][1]), 2, (255, 0, 0), -1)

    # cv2.imwrite('./re_test_th/k_dingwei.jpg', img_dbg)
    # cv2.namedWindow('img_dbg', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_dbg', img_dbg)
    # cv2.waitKey()

    img = cv2.imread(img_path)
    cnt = np.array(l_pos)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype('float32')
    dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    # cv2.imwrite('./re_test_th/image/iii.jpg', warped)

    # result_dict = drow_judge(warped, i)
    src = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    h, w = src.shape
    # print(src.shape)
    if h < w:
        s = cv2.transpose(src)
        src = cv2.flip(s, 0)
    # else:
    #     src = cv2.flip(src, 0)
    # print(src.shape)
    y = np.sum(src, axis=1)
    # print(len(y))
    tmp = scipy.signal.savgol_filter(y, 25, 4)
    tmp_ng = -1 * tmp
    y_blen = int(round(len(tmp) / 2, 0))  # 半长度
    avg_height = round(np.average(tmp_ng[int(round(y_blen - y_blen / 2)): int(round(y_blen + y_blen / 2))].tolist()))  # 中间4

    # 使用find_peak,找阈值内到拐点
    peak_id, peak_property = scipy.signal.find_peaks(-1 * tmp[0:y_blen], height=avg_height, distance=80)
    peak_id2, peak_property2 = scipy.signal.find_peaks(-1 * tmp[y_blen:], height=avg_height, distance=80)
    # print(peak_property2['peak_heights'][0])

    if len(peak_id) != 0:
        y1_id = peak_id[0]
    else:
        y1_id = [0][0]

    if len(peak_id2) != 0:
        y2_id = peak_id2[0]
    else:
        y2_id = [0][0]
    # yyy = np.array(-1*tmp[0:y_blen])
    # peak_height = peak_property['peak_heights']
    # print('peak_freq', peak_id[0])

    y1 = tmp[0:y_blen].tolist()
    y2 = tmp[y_blen:].tolist()

    x1 = [i for i in range(len(y1))]
    x2 = [i for i in range(len(y2))]
    list_xy = [[x1, y1], [x2, y2]]

    list_xy_index = 0
    tc_half_width = []
    for xx, yy in list_xy:

        kneedle_left = KneeLocator(
            xx, yy, curve="concave", direction="decreasing", online=True)
        kneedle_right = KneeLocator(
            xx, yy, curve="concave", direction="increasing", online=True)

        kl_x = list(kneedle_left.all_knees)
        kl_y = [yy[i] for i in kl_x]
        kr_x = list(kneedle_right.all_knees)
        kr_y = [yy[i] for i in kr_x]
        # lfrg_list = [kl_x, kl_y*M, kr_x, kr_y*M]
        # lfrg_list = [kl_x, kl_y]
        # print(lfrg_list)
        try:
            if list_xy_index == 0:
                klx = max([i for i in kl_x if i < peak_id])
                krx = min([i for i in kr_x if i > peak_id])
                full_len = math.sqrt(
                    (klx - krx) ** 2 + (y1[klx] - y1[krx]) ** 2
                )  # 通过左右拐点坐标进行取半峰宽
                tc_half_width.append(round(full_len / 2))
                list_xy_index += 1
            elif list_xy_index == 1:
                klx = max([i for i in kl_x if i < peak_id2])
                krx = min([i for i in kr_x if i > peak_id2])
                full_len = math.sqrt((klx - krx) ** 2 + (y1[klx] - y1[krx]) ** 2)
                tc_half_width.append(round(full_len / 2))

        except:
            tc_half_width.append(0)
    t_width_half, c_width_half = tc_half_width
    y1_result = 1 if t_width_half > 10 else 0  # 判断TC的存在
    y2_result = 1 if c_width_half > 10 else 0
    # print('y1_reslut:', y1_result)
    # print('y2_reslut:', y2_result)
    avg_real_abs = abs(y2[y2_id] + avg_height)
    # print(avg_real_abs)
    dg = y2_id + y_blen - y1_id
    # print('差值为', dg)
    if y1_result == 1 and y2_result == 1:

        if 50 <= dg <= 147 and avg_real_abs > 70:  # 124.5-134
            if y1[y1_id] / y2[y2_id] <= 0.749:
                con = "20mlu ml"
            elif 0.75 <= y1[y1_id] / y2[y2_id] <= 1:
                con = "50mlu ml"
            else:
                con = "2000mlu ml"
            result_val = '+' + ' ' + str(con)
            # print(con, y1v3/y2v3)
        else:
            result_val = '-'

    elif y1_result == 1 and y2_result == 0:

        if 50 <= dg <= 147 and avg_real_abs > 70:
            if y1[y1_id] / y2[y2_id] <= 0.749:
                con = "20mlu ml"
            elif 0.75 <= y1[y1_id] / y2[y2_id] <= 1:
                con = "50mlu ml"
            else:
                con = "2000mlu ml"
            result_val = '+' + ' ' + str(con)
            # print(con, y1v3/y2v3)
        else:
            result_val = '-'

    elif y1_result == 0 and y2_result == 0 or y2_result == 1:

        if 50 <= dg <= 147 and avg_real_abs > 70:
            if y1[y1_id] / y2[y2_id] <= 0.749:
                con = "20mlu ml"
            elif 0.75 <= y1[y1_id] / y2[y2_id] <= 1:
                con = "50mlu ml"
            else:
                con = "2000mlu ml"
            result_val = '+' + ' ' + str(con)
            # print(con, y1v3/y2v3)
        else:
            result_val = '-'

    else:
        result_val = 'NAL'
    result_val = {'HCG': result_val}
    jsondict = json.dumps(result_val)
    print(jsondict)
    # plt.plot(tmp, label='拟合曲线', color='r')
    # # plt.plot(y, label='原走势', color='b')
    # plt.title('blue is orignal trend, red is smooth trend')
    # # plt.savefig('./re_test_th/image/i.png')
    # plt.show()
    # # plt.clf()

    # return result_val, img_dbg


relative_position_photo_skjtj = {

    'j1a': {'rad': 0.48, 'dist': -405.000},
    'j1b': {'rad': 0.44, 'dist': -396.000},
    'j1c': {'rad': 0.81, 'dist': -255.000},
    'j1d': {'rad': 0.75, 'dist': -243.000},
    'j2a': {'rad': -0.01, 'dist': -355.000},
    'j2b': {'rad': 0.04, 'dist': -355.000},
    'j2c': {'rad': -0.01, 'dist': -175.000},
    'j2d': {'rad': 0.09, 'dist': -175.000},
    'j3a': {'rad': 5.86, 'dist': -388.000},
    'j3b': {'rad': 5.82, 'dist': -395.000},
    'j3c': {'rad': -0.78, 'dist': -249.000},
    'j3d': {'rad': -0.74, 'dist': -238.000},
}


# 三合一胶体金
def xinzhi_process_skjtj(posn, img_path, img_dbg):
    SCALE = 1
    # img_dbg = cv2.imread(img_path)
    RADIAN = updata_radian(posn[b'2'], posn[b'6'])  # 计算与水平的偏度
    cv2.circle(img_dbg, posn[b'2'], 5, (255, 0, 0), -1)
    cv2.circle(img_dbg, posn[b'6'], 5, (255, 0, 0), -1)
    cv2.circle(img_dbg, (int((posn[b'6'][0] + posn[b'2'][0]) / 2), int((posn[b'6'][1] + posn[b'2'][1]) / 2)), 5,
               (255, 0, 0), -1)
    l_pos = get_position(img_path, relative_position_photo_skjtj, SCALE, RADIAN, posn[b'2'])  # 中心坐标

    for iii in range(12):
        cv2.circle(img_dbg, (l_pos[iii][0], l_pos[iii][1]), 2, (255, 0, 0), -1)

    # if not os.path.isdir("/usr/local/service/prod/python/debug/pic/pic"):
    #     os.makedirs("/usr/local/service/prod/python/debug/pic/pic")
    if not os.path.isdir('./pic/pic'):
        os.makedirs('./pic/pic')

    v = 1
    l =[]
    for i in l_pos:
        l.append(i)
        if len(l) == 4:

            img = cv2.imread(img_path)
            cnt = np.array(l)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = box.astype('float32')
            dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype='float32')
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (width, height))
            # imagename = "/usr/local/service/prod/python/debug/pic/pic/" + str(v) + ".jpg"
            imagename = "./pic/pic/" + str(v) + ".jpg"
            cv2.imwrite(imagename, warped)
            # cv2.imshow('aa', warped)
            # cv2.waitKey(0)
            l = []
            v += 1

    # for root, dir, files in os.walk("/usr/local/service/prod/python/debug/pic/pic"):
    for root, dir, files in os.walk('./pic/pic'):
        cut_path = []
        for filename in files:
            apath = os.path.join(root, filename)
            cut_path.append(apath)

    p = 1
    result_dict = {}
    cut_path.sort()
    for k in cut_path:
        readcv = cv2.imread(k)
        src = cv2.cvtColor(readcv, cv2.COLOR_BGR2GRAY)
        h, w = src.shape
        # print(src.shape)
        if h < w:
            s = cv2.transpose(src)
            src = cv2.flip(s, 0)
        cv2.namedWindow('img111', cv2.WINDOW_NORMAL)
        cv2.imshow('img111', src)
        cv2.waitKey(0)
        result_dict[p] = drow_judge_sk(src, p)
        p += 1
    jsondict = json.dumps(result_dict)
    print(jsondict)
    return result_dict, img_dbg


def drow_judge_sk(src, p):
    """
    :param src:   图像
    :param p:  名字
    :return: 返回值
    """
    y = np.sum(src, axis=1)
    # print(len(y))
    tmp = scipy.signal.savgol_filter(y, 25, 4)
    tmp_ng = -1 * tmp
    y_blen = int(round(len(tmp) / 2, 0))  # 半长度
    avg_height = round(
        np.average(tmp_ng[int(round(y_blen - y_blen / 2)): int(round(y_blen + y_blen / 2))].tolist()))  # 中间4

    # 使用find_peak,找阈值内到拐点
    peak_id, peak_property = scipy.signal.find_peaks(-1 * tmp[0:y_blen], height=avg_height, distance=80)
    peak_id2, peak_property2 = scipy.signal.find_peaks(-1 * tmp[y_blen:], height=avg_height, distance=80)
    # print(peak_property2['peak_heights'][0])

    if len(peak_id) != 0:
        y1_id = peak_id[0]
    else:
        y1_id = [0][0]

    if len(peak_id2) != 0:
        y2_id = peak_id2[0]
    else:
        y2_id = [0][0]
    # yyy = np.array(-1*tmp[0:y_blen])
    # peak_height = peak_property['peak_heights']
    # print('peak_freq', peak_id[0])

    y1 = tmp[0:y_blen].tolist()
    y2 = tmp[y_blen:].tolist()

    x1 = [i for i in range(len(y1))]
    x2 = [i for i in range(len(y2))]
    list_xy = [[x1, y1], [x2, y2]]

    list_xy_index = 0
    tc_half_width = []
    for xx, yy in list_xy:

        kneedle_left = KneeLocator(
            xx, yy, curve="concave", direction="decreasing", online=True)
        kneedle_right = KneeLocator(
            xx, yy, curve="concave", direction="increasing", online=True)

        kl_x = list(kneedle_left.all_knees)
        kl_y = [yy[i] for i in kl_x]
        kr_x = list(kneedle_right.all_knees)
        kr_y = [yy[i] for i in kr_x]
        # lfrg_list = [kl_x, kl_y*M, kr_x, kr_y*M]
        # lfrg_list = [kl_x, kl_y]
        # print(lfrg_list)
        try:
            if list_xy_index == 0:
                klx = max([i for i in kl_x if i < peak_id])
                krx = min([i for i in kr_x if i > peak_id])
                full_len = math.sqrt(
                    (klx - krx) ** 2 + (y1[klx] - y1[krx]) ** 2
                )  # 通过左右拐点坐标进行取半峰宽
                tc_half_width.append(round(full_len / 2))
                list_xy_index += 1
            elif list_xy_index == 1:
                klx = max([i for i in kl_x if i < peak_id2])
                krx = min([i for i in kr_x if i > peak_id2])
                full_len = math.sqrt((klx - krx) ** 2 + (y1[klx] - y1[krx]) ** 2)
                tc_half_width.append(round(full_len / 2))

        except:
            tc_half_width.append(0)
    t_width_half, c_width_half = tc_half_width
    y1_result = 1 if t_width_half > 10 else 0  # 判断TC的存在
    y2_result = 1 if c_width_half > 10 else 0
    # print('y1_reslut:', y1_result)
    # print('y2_reslut:', y2_result)
    avg_real_abs = abs(y2[y2_id] + avg_height)
    # print(avg_real_abs)
    dg = y2_id + y_blen - y1_id
    # print('差值为', dg)
    if y1_result == 1 and y2_result == 1:

        if 50 <= dg <= 147 and avg_real_abs > 70:
            # if y1[y1_id] / y2[y2_id] <= 0.749:
            #     con = "20mlu ml"
            # elif 0.75 <= y1[y1_id] / y2[y2_id] <= 1:
            #     con = "50mlu ml"
            # else:
            #     con = "2000mlu ml"
            result_val = '+'
            # print(con, y1v3/y2v3)
        else:
            result_val = '-'

    elif y1_result == 1 and y2_result == 0:

        if 50 <= dg <= 147 and avg_real_abs > 70:
            # if y1[y1_id] / y2[y2_id] <= 0.749:
            #     con = "20mlu ml"
            # elif 0.75 <= y1[y1_id] / y2[y2_id] <= 1:
            #     con = "50mlu ml"
            # else:
            #     con = "2000mlu ml"
            result_val = '+'
            # print(con, y1v3/y2v3)
        else:
            result_val = '-'

    elif y1_result == 0 and y2_result == 0 or y2_result == 1:

        if 50 <= dg <= 147 and avg_real_abs > 70:
            # if y1[y1_id] / y2[y2_id] <= 0.749:
            #     con = "20mlu ml"
            # elif 0.75 <= y1[y1_id] / y2[y2_id] <= 1:
            #     con = "50mlu ml"
            # else:
            #     con = "2000mlu ml"
            result_val = '+'
            # print(con, y1v3/y2v3)
        else:
            result_val = '-'

    else:
        result_val = 'NAL'
    print(result_val)
    # result_val = {'1': result_val}
    # plt.plot(tmp, label='拟合曲线', color='r')
    # # plt.plot(y, label='原走势', color='b')
    # plt.title('blue is orignal trend, red is smooth trend')
    # # plt.savefig('./re_test_th/image/i.png')
    # plt.show()
    # plt.clf()
    # return result_val