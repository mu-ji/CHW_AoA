import numpy as np
import matplotlib.pyplot as plt
import AoA_algorithm

def complete_reference_phase_data(reference_ant_data):
    diff = reference_ant_data[1:]-reference_ant_data[:-1]
    for i in range(len(diff)):
        if abs(diff[i]) >= 100:
            diff[i] = 0
    slope = np.mean(diff[diff != 0])
    for i in range(len(reference_ant_data)):
        if reference_ant_data[i] != 0:
            continue
        else:
            next_phase = reference_ant_data[i-1] + slope
            if next_phase > 201:
                next_phase = next_phase - 402
            reference_ant_data[i] = next_phase
    return reference_ant_data, slope/0.125

def find_best_intercept(x, y, slope):
    n = len(x)
    A = np.vstack([x, np.ones(n)]).T
    b = np.sum(y - np.multiply(x,slope)) / n
    return b
    
def complete_other_phase_data(ant_phase_data,reference_slope):
    nonzero_indices = np.nonzero(ant_phase_data)[0]
    min_idx = nonzero_indices[0]
    max_idx = nonzero_indices[-1]
    x = [i*0.125 for i in range(min_idx,max_idx)]
    y = ant_phase_data[min_idx:max_idx]
    for i in range(len(y)-1):
        if y[i]-y[i+1] >= 200:
            y[i+1] = y[i+1] + 402

    intercept = find_best_intercept(x,y,reference_slope)
    first_index = min_idx
    last_index = min_idx + 1
    ant_phase_data[first_index] = reference_slope*first_index*0.125 + intercept
    ant_phase_data[last_index] = reference_slope*last_index*0.125 + intercept
    while first_index != 0 or last_index != (len(ant_phase_data)-1):
        if first_index != 0:
            last_phase = ant_phase_data[first_index] - 0.125*reference_slope
            if last_phase <= -201:
                last_phase = last_phase + 402
            ant_phase_data[first_index-1] = last_phase
            first_index = first_index - 1
        if last_index != (len(ant_phase_data)-1):
            next_phase = ant_phase_data[last_index] + 0.125*reference_slope
            if next_phase > 201:
                next_phase = next_phase - 402
            ant_phase_data[last_index+1] = next_phase
            last_index = last_index + 1
    
    return ant_phase_data

def release_jump(diff):
    for i in range(len(diff)):
        if abs(diff[i]) > 201:
            if diff[i] < -201:
                diff[i] = diff[i] + 402
            if diff[i] > 201:
                diff[i] = diff[i] - 402
    return diff


def two_ant_cal_angle(mean_diff, wave_length, d):
    arccos = (mean_diff/402)*wave_length/d
    if arccos > 1:
        arccos = 1
    elif arccos < -1:
        arccos = -1
    return np.arccos(arccos)/3.14*180

def find_x_angle_in_row(antenna_array_row_1):
    antenna_array_row_1 = np.exp(1j * (antenna_array_row_1))
    array = np.array([0, 0.375, 0.75, 1.1125])#np.linspace(0,(4-1)/2,4)
    Angles = np.linspace(-np.pi/2,np.pi/2,360)
    Cov_row_1 = antenna_array_row_1@antenna_array_row_1.conj().transpose()

    DoAsMUSIC, psindB = AoA_algorithm.music(Cov_row_1,1,4,array,Angles)
    try:
        return Angles[DoAsMUSIC][0]/3.14*180
    except:
        return -180


def array_ant_cal_x_angle(antenna_phase_array, wave_length, d):
    '''
    思路如下
    一个4*4的天线阵列，为了计算在水平方向上的角度，每行天线由MUSIC先计算一个角度，四行天线获得四个角度再平均。
    '''
    x_angle_row_1 = 0
    x_angle_row_2 = 0
    x_angle_row_3 = 0
    x_angle_row_4 = 0

    antenna_array_row_1 = antenna_phase_array[0,:,:]
    antenna_array_row_2 = antenna_phase_array[1,:,:]
    antenna_array_row_3 = antenna_phase_array[2,:,:]
    antenna_array_row_4 = antenna_phase_array[3,:,:]        #shape = 4*320
    x_angle_row_1 = find_x_angle_in_row(antenna_array_row_1)
    x_angle_row_2 = find_x_angle_in_row(antenna_array_row_2)
    x_angle_row_3 = find_x_angle_in_row(antenna_array_row_3)
    x_angle_row_4 = find_x_angle_in_row(antenna_array_row_4)
    angle_array = [x_angle_row_1,x_angle_row_2,x_angle_row_3,x_angle_row_4]

    return angle_array


