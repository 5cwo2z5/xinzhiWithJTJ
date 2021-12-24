import numpy as np

def std(lst, x):
    return np.sqrt(np.mean(abs(np.array(lst) - x)**2))

def make_int_aver(lst):
    return round(sum(lst)/len(lst))

def color_approximation(src, dst):
    src, dst = map(np.array, (src, dst))
    x = np.dot(src, dst)/(np.sqrt(sum(src**2))*np.sqrt(sum(dst**2)))
    r = np.arccos(x)*(180/np.pi)
    return r

def pos2box(pos_list, sizex, sizey):
    box_list = []
    for pos in pos_list:
        box = []
        box.extend([pos[0]-(sizex//2), pos[1]-(sizey//2), sizex, sizey])
        box_list.append(box)
    return box_list

def read_boxes_rgb(img, boxes, color_shift):

    def get_rgb_in_square(im, box):
        posx, posy, sizex, sizey = box
        lst = []
        for y in range(posy, posy+sizey):
            for x in range(posx, posx+sizex):
                b, g, r = im[y, x]
                rgb = (int(r), int(g), int(b))
                lst.append(rgb)
        return lst 

    def get_main_color(im, box):
        color_lst = get_rgb_in_square(im, box)
        if color_lst == 0:
            print('No color data.')
            return 0
        r, g, b = [[item[i] for item in color_lst] for i in range(len(color_lst[0]))]

        rmax, gmax, bmax = map(max, (r, g, b))
        std_rm = std(r, rmax)
        std_gm = std(g, gmax)
        std_bm = std(b, bmax)

        ravg, gavg, bavg = map(make_int_aver, (r, g, b))

        std_ra = std(r, ravg)
        std_ga = std(g, gavg)
        std_ba = std(b, bavg)

        rmain = rmax if std_ra > std_rm else ravg
        gmain = gmax if std_ga > std_gm else gavg
        bmain = bmax if std_ba > std_bm else bavg

        return [rmain, gmain, bmain]

    rgbs = []
    for box in boxes:
        c = get_main_color(img, box)

        c[0] *= color_shift[0]
        c[1] *= color_shift[1]
        c[2] *= color_shift[2]

        if c[0] < 0: c[0] = 0
        if c[1] < 0: c[1] = 0
        if c[2] < 0: c[2] = 0
        if c[0] > 255: c[0] = 255
        if c[1] > 255: c[1] = 255
        if c[2] > 255: c[2] = 255

        rgbs.append(c)
    return list(rgbs)
