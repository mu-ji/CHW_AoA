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
antenna_id_list = [i for i in range(4)]
antenna_diff_list = []

while True:
    byte  = ser.read(1)        
    rawFrame += byte
    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 2312:
            received_data = rawFrame[:2304]
            num_samples = 576

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
            '''
            I = mag_data * np.cos(phase_data)  # 同相分量
            Q = mag_data * np.sin(phase_data)   # 正交分量

            plt.figure()
            plt.plot([i*0.125 for i in range(576)], I, label='I')
            plt.plot([i*0.125 for i in range(576)], Q, label='Q')
            plt.show()
            '''
            offset_phase_data = np.zeros_like(phase_data)

            mag_data = mag_data.astype(np.float32)
            
            #phase_data_diff = np.diff(phase_data)

            antenna0_data_list = []
            i = 72
            reference_ant_data = np.zeros_like(phase_data)
            reference_ant_data[0:64] = phase_data[0:64]
            reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)

            while i+8 < 576:
                antenna0_data = np.zeros_like(phase_data)
                antenna0_data[i:i+8] = phase_data[i:i+8]
                antenna0_data = AoA_cal_angle.complete_other_phase_data(antenna0_data,reference_slope)
                antenna0_data_list.append(antenna0_data[72:80])

                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax2.plot([i*0.125 for i in range(576)],mag_data, marker='*')
                ax1.plot([i*0.125 for i in range(576)],phase_data, marker='*')
                ax1.plot([i*0.125 for i in range(576)],antenna0_data)

                k = 8
                flag = True
                while k < 72:
                    ax1_switch_slot = plt.Rectangle(xy=(8*k*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'r')
                    ax2_switch_slot = plt.Rectangle(xy=(8*k*0.125 , 100), width=1, height=200, alpha=0.2, angle=0.0, color = 'r')
                    ax1.add_patch(ax1_switch_slot)
                    ax2.add_patch(ax2_switch_slot)
                    if flag:
                        ax1_sample_slot = plt.Rectangle(xy=(8*(k+1)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'g')
                        ax2_sample_slot = plt.Rectangle(xy=(8*(k+1)*0.125 , 100), width=1, height=200, alpha=0.2, angle=0.0, color = 'g')
                        flag = False
                    else:
                        ax1_sample_slot = plt.Rectangle(xy=(8*(k+1)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'y')
                        ax2_sample_slot = plt.Rectangle(xy=(8*(k+1)*0.125 , 100), width=1, height=200, alpha=0.2, angle=0.0, color = 'y')
                        flag = True
                    ax1.add_patch(ax1_sample_slot)
                    ax2.add_patch(ax2_sample_slot)
                    k = k + 2
                plt.show()
                i = i + 16

            antenna0_data_array = np.array(antenna0_data_list)
            antenna_diff_list.append(antenna0_data_array)

            #print(phase_diff.shape)
            #fig, ax = plt.subplots(figsize=(10, 6))
            #plt.violinplot(phase_diff.T,showmeans=True)
            #plt.show()
            '''
            antenna_diff_once = []
            for i in antenna_id_list:
                antenna_id = i

                sample_1 = phase_data[64+(antenna_id+1)*8:64+(antenna_id+1)*8 + 8]
                sample_2 = phase_data[64+(antenna_id+1)*8 + 64:64+(antenna_id+1)*8 + 8 + 64]

                phase_diff = sample_2 - sample_1
                #print('befor:',phase_diff)
                phase_diff = AoA_cal_angle.release_jump(phase_diff)
                #print('after:',phase_diff)
                antenna_diff_once.append(phase_diff)
            antenna_diff_list.append(antenna_diff_once)
            #fig, ax = plt.subplots(figsize=(10, 6))
            #plt.violinplot(antenna_diff_once, showmeans=True)
            #plt.show()
            '''

            if np.array(antenna_diff_list).shape == (100, 16, 8):
                np.save('switch_phase_different/0005_72_0.5_90_90_1.npy', np.array(antenna_diff_list))
                break

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

