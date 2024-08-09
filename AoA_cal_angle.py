import numpy as np
import matplotlib.pyplot as plt

def two_ant_cal_angle(mean_diff, wave_length, d):
    arccos = (mean_diff/402)*wave_length/d
    if arccos > 1:
        arccos = 1
    elif arccos < -1:
        arccos = -1
    #if arccos < 0:
    #    #print('angle<0:',np.arccos((mean_diff/402)*wave_length/d)/3.14*180)
    #    return 180 - np.arccos(arccos)/3.14*180
    #else:
    return np.arccos(arccos)/3.14*180