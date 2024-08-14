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

times = 10
iteration = 0
#while iteration < times:
while True:
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
                #print(phase)
                #print((received_data[4*i+2] << 8) | received_data[4*i+3])
                #phase_data[i] = (received_data[4*i+2] << 8) | received_data[4*i+3]
                phase_data[i] = phase[0]
                mag_data[i] = mag[0]

            phase_data = phase_data.astype(np.float32)
            mag_data = mag_data.astype(np.float32)
            
            #phase_data_diff = np.diff(phase_data)

            iteration = iteration + 1
            
            plt.figure()
            i = 8
            plt.plot([i*0.125 for i in range(320)],phase_data, marker='*')
            plt.plot([8*i*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+1)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+2)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+3)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+4)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+5)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+6)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+7)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+8)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+9)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+10)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+11)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+12)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+13)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+14)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+15)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+16)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+17)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+18)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+19)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+20)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+21)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+22)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+23)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+24)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+25)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+26)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+27)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+28)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+29)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+30)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+31)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+32)*0.125]*2, [-201,201],c = 'b')
            plt.legend()
            plt.show()

            np.save('phase_data.npy',phase_data)

            reference_ant_data = np.zeros_like(phase_data)
            ant1_data = np.zeros_like(phase_data)
            ant2_data = np.zeros_like(phase_data)
            ant3_data = np.zeros_like(phase_data)

            reference_ant_data[0:64] = phase_data[0:64]
            ant1_data[72:80] = phase_data[72:80]
            ant2_data[88:96] = phase_data[88:96]
            ant3_data[104:112] = phase_data[104:112]

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
            ant1_data = complete_other_phase_data(ant1_data,reference_slope)
            ant2_data = complete_other_phase_data(ant2_data,reference_slope)
            ant3_data = complete_other_phase_data(ant3_data,reference_slope)

            diff1 = ant1_data[72:80] - ant3_data[72:80]
            diff2 = ant2_data[88:96] - ant3_data[88:96]

            #print('diff1:',diff1)
            #print('diff2:',diff2)

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


            #print('mean_phase_diff_1:', mean_phase_diff_1)
            #print('mean_phase_diff_2:', mean_phase_diff_2)

            wave_length = 0.125 # m
            antenna_interval = 0.0375 # m

            angle1 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_1,wave_length,antenna_interval)
            angle2 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_2,wave_length,2*antenna_interval)

            #print('angle1:',angle1)
            #print('angle2:',angle2)

            if np.std((angle1,angle2)) >= 20:
                print('not stable')
            else:
                print('estimate angle:', np.mean((angle1,angle2)))
            
            '''
            plt.figure()
            i = 8
            plt.plot([8*i*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+1)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+2)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+3)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+4)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+5)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+6)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+7)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+8)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+9)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+10)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+11)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+12)*0.125]*2, [-201,201],c = 'b')
            plt.plot([i*0.125 for i in range(160)],phase_data, marker='*')
            plt.plot([i*0.125 for i in range(160)],reference_ant_data,c = 'r',label = 'ANT1_1 (ref)')
            plt.plot([i*0.125 for i in range(160)],ant1_data,c = 'g',label = 'ANT1_2')
            plt.plot([i*0.125 for i in range(160)],ant2_data,c = 'y',label = 'ANT1_3')
            plt.plot([i*0.125 for i in range(160)],ant3_data,c = 'pink',label = 'ANT1_1')
            plt.legend()
            plt.show()
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

