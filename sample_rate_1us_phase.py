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
ser = serial.Serial('COM11', 115200)

import cmath

SPEED_OF_LIGHT  = 299792458

rawFrame = []

diff_list = []

times = 10
iteration = 0
#while iteration < times:
antenna_id_list = [i for i in range(4)]
antenna_diff_list = []

def expend_samples(phase_data, time_id, phase_shift_per_us):
    expend_line = np.zeros_like(phase_data)
    expend_line[time_id] = phase_data[time_id]
    min_id = time_id
    max_id = time_id
    while min_id > 0 or max_id < len(phase_data):
        if min_id > 0:
            expend_line[min_id - 1] = expend_line[min_id] - phase_shift_per_us
            min_id = min_id - 1
        if max_id < len(phase_data):
            expend_line[min_id + 1] = expend_line[min_id] + phase_shift_per_us
            min_id = min_id + 1

    return expend_line

def three_antenna_cal_angle(ant0_data, ant1_data, ant2_data, wave_length, d):
    phase_diff = np.diff([np.mean(ant0_data), np.mean(ant1_data), np.mean(ant2_data)])
    print('phase_diff:', phase_diff)
    if np.std(phase_diff) > 10:
        return 'illegal_measurement'
    else:
        mean_diff = np.mean(phase_diff)
    print('mean_diff:', mean_diff)
    arccos = (mean_diff/402)*wave_length/d
    if arccos > 1:
        arccos = 1
    elif arccos < -1:
        arccos = -1
    return np.arccos(arccos)/3.14*180

def release_jump(ant_data):
    for i in range(len(ant_data)):
        if ant_data[i] < 0:
            ant_data[i] = ant_data[i] + 403
    return ant_data

num_samples = 88
while True:
    byte  = ser.read(1)        
    rawFrame += byte
    #print(len(rawFrame))
    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 4*num_samples+8:
            received_data = rawFrame[:4*num_samples]
            num_samples = 88

            phase_data = np.zeros(num_samples, dtype=np.int16)
            mag_data = np.zeros(num_samples, dtype=np.int16)
            for i in range(num_samples):
                (phase) = struct.unpack('>h', bytes(received_data[4*i+2:4*i+4]))
                (mag) = struct.unpack('>h', bytes(received_data[4*i:4*i+2]))
                #print(phase)
                #print((received_data[4*i+2] << 8) | received_data[4*i+3])
                #phase_data[i] = (received_data[4*i+2] << 8) | received_data[4*i+3]
                phase_data[i] = phase[0]
                mag_data[i] = mag[0]

            phase_data = phase_data.astype(np.float32)
            mag_data = mag_data.astype(np.float32)
            
            reference_phase = np.zeros_like(phase_data)
            ant0_phase = np.zeros_like(phase_data)
            ant1_phase = np.zeros_like(phase_data)
            ant2_phase = np.zeros_like(phase_data)
            ant3_phase = np.zeros_like(phase_data)

            reference_phase = phase_data[:8]
            ant0_phase = phase_data[9:88:8]
            ant1_phase = phase_data[11:88:8]
            ant2_phase = phase_data[13:88:8]
            ant3_phase = phase_data[15:88:8]

            reference_phase,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_phase)
            ant0_phase = AoA_cal_angle.complete_other_phase_data(ant0_phase,reference_slope)
            ant1_phase = AoA_cal_angle.complete_other_phase_data(ant1_phase,reference_slope)
            ant2_phase = AoA_cal_angle.complete_other_phase_data(ant2_phase,reference_slope)
            ant3_phase = AoA_cal_angle.complete_other_phase_data(ant3_phase,reference_slope)

            def steering_vector(alpha):
                j = 1j  # 复数单位
                return np.array([1, cmath.exp(-j * 2 * np.pi * 2.4e9 * (0.0375*np.sin(alpha)/SPEED_OF_LIGHT)), cmath.exp(-j * 2 * np.pi * 2.4e9 * 2*(0.0375*np.sin(alpha)/SPEED_OF_LIGHT))])

            def DoA_algorithm(ant0_phase, ant1_phase, ant2_phase):

                ant1_theta = np.mean(AoA_cal_angle.release_jump(ant1_phase - ant0_phase))/402*2*np.pi
                ant2_theta = np.mean(AoA_cal_angle.release_jump(ant2_phase - ant0_phase))/402*2*np.pi
                print(ant1_theta, ant2_theta)
                if ant2_theta/ant1_theta >= 3:
                    ant2_theta - ant2_theta - 2*np.pi
                ant0_theta = 0

                received_signal = np.array([cmath.exp(1j*ant0_theta), cmath.exp(1j*ant1_theta), cmath.exp(1j*ant2_theta)])
                angle_list = [np.radians(i) for i in range(-90, 90)]
                y_alpha_list = []
                for alpha in angle_list:
                    y_alpha = steering_vector(alpha)[0]*received_signal[0] + steering_vector(alpha)[1]*received_signal[1] + steering_vector(alpha)[2]*received_signal[2]
                    y_alpha_list.append(y_alpha)

                #plt.plot([i for i in range(-90, 90)], y_alpha_list)
                #plt.show()
                return [i for i in range(-90, 90)][np.argmax(np.array(y_alpha_list))]
            
            angle = DoA_algorithm(ant0_phase, ant1_phase, ant2_phase)
            print('DoA:', angle)
            '''
            def cal_signal_diraction(ant0_I_mean, ant0_Q_mean, ant3_I_mean, ant3_Q_mean):
                phase_diff_0_3 = calculate_angle(ant0_I_mean, ant0_Q_mean, ant3_I_mean, ant3_Q_mean)
                #print(phase_diff_0_3)
                if calculate_angle(ant3_I_mean, ant3_Q_mean, rotate_vector(ant0_I_mean, ant0_Q_mean, phase_diff_0_3)[0], rotate_vector(ant0_I_mean, ant0_Q_mean, phase_diff_0_3)[1]) < 1:
                    arccos = (phase_diff_0_3/(2*np.pi))*0.125/0.0375
                    #print('arccos:', arccos)
                    if arccos > 1:
                        arccos = 1
                    elif arccos < -1:
                        arccos = -1
                    return np.arccos(arccos)/np.pi*180
                else:
                    arccos = (-phase_diff_0_3/(2*np.pi))*0.125/0.0375
                    if arccos > 1:
                        arccos = 1
                    elif arccos < -1:
                        arccos = -1
                    return np.arccos(arccos)/np.pi*180

            #angle_0_1 = cal_signal_diraction(ant0_I_mean, ant0_Q_mean, ant1_I_mean, ant1_Q_mean)
            #angle_2_3 = cal_signal_diraction(ant2_I_mean, ant2_Q_mean, ant3_I_mean, ant3_Q_mean)
            #print(angle_0_1)
            #print(angle_2_3)
            #print('x_angle:', (angle_0_1+angle_2_3)/2)
            '''

            try:
                response_rssi = bytes(rawFrame[-8:-4])
                response_rssi = int(response_rssi.decode('utf-8'))
                #print(iteration)
                print(response_rssi)
                #print('packet_number:',rawFrame[-4])
                #print('-------------------------------')

            except:
                rawFrame = []
                continue
            
        rawFrame = []

