import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii

import matplotlib.pyplot as plt
from math import pi, atan2, sqrt
from scipy.linalg import eig

import AoA_algorithm
import AoA_cal_angle
import AoA_filter

def update_earlier_measurement(data_array, new_data):

    if data_array.shape[0] < 10:
        # data_array未满,直接添加新数据
        data_array = np.vstack((data_array, new_data))
    else:
        # data_array已满,删除最早的数据并添加新数据
        data_array = np.delete(data_array, 0, axis=0)
        data_array = np.vstack((data_array, new_data))
    
    return data_array

raw_angle_list = []
KF_angle_list = []

def angle_estimation(true_angle):
    ser = serial.Serial('COM8', 115200)
    rawFrame = []

    x_earlier_measurement = np.zeros((0,4))
    kalman_filter_x = AoA_filter.Kalman_Filter()

    #while iteration < times:
    while True:
        if len(raw_angle_list) >= 100:
            break
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-3:]==[255, 255, 255]:
            if len(rawFrame) == 1288:
                received_data = rawFrame[:1288]
                num_samples = 320

                phase_data = np.zeros(num_samples, dtype=np.int16)
                mag_data = np.zeros(num_samples, dtype=np.int16)
                for i in range(num_samples):
                    (phase) = struct.unpack('>h', bytes(received_data[4*i+2:4*i+4]))
                    (mag) = struct.unpack('>h', bytes(received_data[4*i:4*i+2]))
                    phase_data[i] = phase[0]
                    mag_data[i] = mag[0]
                phase_data = phase_data.astype(np.float32)
                mag_data = mag_data.astype(np.float32)

                antenna_phase_array = np.zeros((4,4,8))

                reference_ant_data = np.zeros_like(phase_data)
                antenna_phase_array = np.zeros((4,4,len(phase_data)))
                reference_ant_data[0:64] = phase_data[0:64]
                reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)
                k = 72
                for i in range(4):
                    for j in range(4):
                        #if i == 0 and j == 0:
                        #    antenna_phase_array[i][j] = AoA_cal_angle.complete_other_phase_data(phase_data[312:320],reference_slope)
                        temple_phase = np.zeros_like(phase_data)
                        temple_phase[k:k+8] = phase_data[k:k+8]
                        antenna_phase_array[i][j] = AoA_cal_angle.complete_other_phase_data(temple_phase,reference_slope)
                        #antenna_phase_array[i][j] = phase_data[k:k+8]
                        k = k + 16

                wave_length = 0.125 # m
                antenna_interval = 0.0375 # m
                x_angle_array = AoA_cal_angle.array_ant_cal_x_angle_array(antenna_phase_array, wave_length, antenna_interval)
                
                raw_angle = AoA_filter.mean_filter(x_angle_array)

                x_earlier_measurement = update_earlier_measurement(x_earlier_measurement, x_angle_array)
                kalman_filter_x.update_R(x_earlier_measurement)
                kalman_filter_x.predict()
                kalman_filter_x.update(x_angle_array)
                KF_angle = kalman_filter_x.X[0][0]

                raw_angle_list.append(raw_angle)
                KF_angle_list.append(KF_angle)

            rawFrame = []
    np.save('Indoor_experiment_result/raw_angle_{}.npy'.format(true_angle), np.array(raw_angle_list))
    np.save('Indoor_experiment_result/KF_angle_{}.npy'.format(true_angle), np.array(KF_angle_list))
    return raw_angle_list, KF_angle_list

true_angle = 160
raw_angle_list, KF_angle_list = angle_estimation(true_angle)

raw_angle_list = np.load('Indoor_experiment_result/raw_angle_{}.npy'.format(true_angle))
KF_angle_list = np.load('Indoor_experiment_result/KF_angle_{}.npy'.format(true_angle))

plt.figure()
plt.plot([i for i in range(len(raw_angle_list))], raw_angle_list, label = 'raw data')
plt.plot([i for i in range(len(KF_angle_list))], KF_angle_list, label = 'KF data')
plt.legend()
plt.grid()
plt.show()
