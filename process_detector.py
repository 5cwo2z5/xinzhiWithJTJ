import sys
import cv2
import math
import json
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from algorithm_detection_xinzhi import xinzhi_process
from algorithm_detection_xinzhi_jtj import xinzhi_process_jtj
from block_detector_xz import qr_detector
from algorithm_get14results import strip_process
from k_algorithm import xinzhi_process_dkjtj

def main():
    STEP_SIZE = 30
    SQUARE_SIZE = 200

    try:
        img_path = sys.argv[1]
        imgcv = cv2.imread(img_path)
        posn = qr_detector(imgcv, STEP_SIZE, SQUARE_SIZE)
        # print(posn)
        if len(posn) == 2:
            if (list(posn.keys()) == [b'2', b'4'] or list(posn.keys()) == [b'4', b'2']):
                xinzhi_process(posn)
            elif (list(posn.keys()) == [b'6', b'5'] or list(posn.keys()) == [b'5', b'6']):
                xinzhi_process_jtj(posn)
            elif (list(posn.keys()) == [b'5', b'2'] or list(posn.keys()) == [b'2', b'5']):
                xinzhi_process_dkjtj(posn, img_path)
        elif len(posn) == 0:
            strip_process()
        else:
            print("error")
    except Exception as e:
        print('error')

if __name__ == "__main__":
    main()
