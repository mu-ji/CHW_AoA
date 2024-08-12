'''
This script try to use mean phse different to estimate the angle of arrival
'''

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

SPEED_OF_LIGHT  = 299792458
frequency = 16000000

rawFrame = []

diff_list = []

number_of_measurement = 10
ant_ref_phase_array = np.zeros((number_of_measurement,160))
ant2_phase_array = np.zeros((number_of_measurement,160))
ant3_phase_array = np.zeros((number_of_measurement,160))
ant4_phase_array = np.zeros((number_of_measurement,160))

times = 0
#while iteration < times:
while True:
    byte  = ser.read(1)        
    rawFrame += byte
    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 648:
            received_data = rawFrame[:640]
            num_samples = 160

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
            

            reference_ant_data = np.zeros_like(phase_data)
            ant2_data = np.zeros_like(phase_data)
            ant3_data = np.zeros_like(phase_data)
            ant4_data = np.zeros_like(phase_data)

            reference_ant_data[0:64] = phase_data[0:64]
            ant2_data[72:80] = phase_data[72:80]
            ant3_data[88:96] = phase_data[88:96]
            ant4_data[104:112] = phase_data[104:112]

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

            reference_ant_data,reference_slope = complete_reference_phase_data(reference_ant_data)
            ant2_data = complete_other_phase_data(ant2_data,reference_slope)
            ant3_data = complete_other_phase_data(ant3_data,reference_slope)
            ant4_data = complete_other_phase_data(ant4_data,reference_slope)
            
            ant_ref_phase_array[times,:] = reference_ant_data
            
            ant2_phase_array[times,:] = ant2_data
            ant3_phase_array[times,:] = ant3_data
            ant4_phase_array[times,:] = ant4_data
            times += 1
            if times == number_of_measurement:
                ant_ref_mean = np.mean(ant_ref_phase_array, axis=0)
                ant_2_mean = np.mean(ant2_phase_array, axis=0)
                ant_3_mean = np.mean(ant3_phase_array, axis=0)
                ant_4_mean = np.mean(ant4_phase_array, axis=0)

                diff1 = ant_2_mean - ant_ref_mean
                diff2 = ant_3_mean - ant_ref_mean
                diff3 = ant_4_mean - ant_ref_mean

                def release_jump(diff):
                    for i in range(len(diff)):
                        if abs(diff[i]) > 201:
                            if diff[i] < -201:
                                diff[i] = diff[i] + 402
                            if diff[i] > 201:
                                diff[i] = diff[i] - 402
                    return diff
                
                mean_phase_diff_1 = np.mean(release_jump(diff1))
                mean_phase_diff_2 = np.mean(release_jump(diff2))
                mean_phase_diff_3 = np.mean(release_jump(diff3))

                print('mean_phase_diff_1:', mean_phase_diff_1)
                print('mean_phase_diff_2:', mean_phase_diff_2)
                print('mean_phase_diff_3:', mean_phase_diff_3)
                wave_length = 0.125 # m
                antenna_interval = 0.0375 # m


                    
                angle1 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_1,wave_length,antenna_interval)
                angle2 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_2,wave_length,2*antenna_interval)
                angle3 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_3,wave_length,3*antenna_interval)

                print('angle1:',angle1)
                print('angle2:',angle2)
                print('angle3:',angle3)
                print('mean_angle:',(angle1+angle2+angle3)/3)
                
                plt.figure()

                plt.plot([i*0.125 for i in range(160)], ant_ref_phase_array[0,:], label = '0')
                plt.plot([i*0.125 for i in range(160)], ant_ref_phase_array[1,:], label = '1')
                plt.plot([i*0.125 for i in range(160)], ant_ref_phase_array[2,:], label = '2')
                plt.legend()
                plt.show()
                times = 0

        
            
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

