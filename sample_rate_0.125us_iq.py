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

num_samples = 704
while True:
    byte  = ser.read(1)        
    rawFrame += byte
    #print(len(rawFrame))
    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 4*num_samples+8:
            received_data = rawFrame[:4*num_samples]
            num_samples = 704

            I_data = np.zeros(num_samples, dtype=np.int16)
            Q_data = np.zeros(num_samples, dtype=np.int16)
            for i in range(num_samples):
                (I) = struct.unpack('>h', bytes(received_data[4*i+2:4*i+4]))
                (Q) = struct.unpack('>h', bytes(received_data[4*i:4*i+2]))
                #print(phase)
                #print((received_data[4*i+2] << 8) | received_data[4*i+3])
                #phase_data[i] = (received_data[4*i+2] << 8) | received_data[4*i+3]
                I_data[i] = I[0]
                Q_data[i] = Q[0]

            I_data = I_data.astype(np.float32)
            Q_data = Q_data.astype(np.float32)

            '''
            plt.figure()
            antenna_samples = (num_samples-64)/64     #3 is number of antennas

            
            ax1 = plt.subplot(231)
            i = 0
            while i < 64:
                ax1.plot([0,I_data[i]], [0, Q_data[i]], label='{}'.format(i))
                i = i+1
            plt.legend()
            #after the while, i = 8 which means switch slot
            
            ax2 = plt.subplot(232)      #ax2 is the phase calculated by IQ data
            ax2.scatter([i for i in range(704)], [64*np.arctan(Q_data[i]/I_data[i]) for i in range(704)])
            plt.legend()

            i = 72
            ax3 = plt.subplot(233)      # ax3 is the pattern of antenna 0
            ax3.set_title('antenna 0 data')
            plt.legend()
            ax4 = plt.subplot(234)      # ax4 is the pattern of antenna 1
            ax4.set_title('antenna 1 data')
            plt.legend()
            ax5 = plt.subplot(235)      # ax5 is the pattern of antenna 2
            ax5.set_title('antenna 2 data')
            plt.legend()
            ax6 = plt.subplot(236)
            ax6.set_title('antenna 3 data')
            
            times = 1
            while i < num_samples and times < antenna_samples:
                for sample_times in range(8):
                    ax3.plot([0,I_data[i+sample_times]], [0, Q_data[i+sample_times]], label='{}'.format(i))
                i = i + 16
                for sample_times in range(8):
                    ax4.plot([0,I_data[i+sample_times]], [0, Q_data[i+sample_times]], label='{}'.format(i))
                i = i + 16
                for sample_times in range(8):
                    ax5.plot([0,I_data[i+sample_times]], [0, Q_data[i+sample_times]], label='{}'.format(i))
                i = i + 16
                for sample_times in range(8):
                    ax6.plot([0,I_data[i+sample_times]], [0, Q_data[i+sample_times]], label='{}'.format(i))
                i = i + 16
                times = times + 1

            plt.legend()
            
            plt.show()
            '''
            reference_I = I_data[:64]
            reference_Q = Q_data[:64]
            
            ant0_I = np.zeros(10,)
            ant0_Q = np.zeros(10,)
            ant1_I = np.zeros(10,)
            ant1_Q = np.zeros(10,)
            ant2_I = np.zeros(10,)
            ant2_Q = np.zeros(10,)
            ant3_I = np.zeros(10,)
            ant3_Q = np.zeros(10,)

            def normalization(ant_I, ant_Q):
                # 计算幅度
                amplitude = np.sqrt(ant_I**2 + ant_Q**2)
                # 归一化 I 和 Q 分量
                norm_I = ant_I / amplitude
                norm_Q = ant_Q / amplitude
                return norm_I, norm_Q

            def ant_IQ_norm(ant_I_array, ant_Q_array):
                for i in range(len(ant_I_array)):
                    ant_I_array[i], ant_Q_array[i] = normalization(ant_I_array[i], ant_Q_array[i])
                return ant_I_array, ant_Q_array
            
            # normalize three antenna IQ data, only consider phase information

            for i in range(10):
                ant0_I_sample, ant0_Q_sample = ant_IQ_norm(I_data[72+64*i:72+64*i+8], Q_data[72+64*i:72+64*i+8])
                ant0_I[i] = np.mean(ant0_I_sample)
                ant0_Q[i] = np.mean(ant0_Q_sample)

                ant1_I_sample, ant1_Q_sample = ant_IQ_norm(I_data[88+64*i:88+64*i+8], Q_data[88+64*i:88+64*i+8])
                ant1_I[i] = np.mean(ant1_I_sample)
                ant1_Q[i] = np.mean(ant1_Q_sample)

                ant_I, ant_Q = ant_IQ_norm(I_data[104+64*i:104+64*i+8], Q_data[104+64*i:104+64*i+8])
                ant2_I[i] = np.mean(ant_I)
                ant2_Q[i] = np.mean(ant_Q)

                ant_I, ant_Q = ant_IQ_norm(I_data[120+64*i:120+64*i+8], Q_data[120+64*i:120+64*i+8])
                ant3_I[i] = np.mean(ant_I)
                ant3_Q[i] = np.mean(ant_Q)

            reference_I, reference_Q = ant_IQ_norm(reference_I, reference_Q)

            ant0_I, ant0_Q = ant_IQ_norm(ant0_I, ant0_Q)
            ant1_I, ant1_Q = ant_IQ_norm(ant1_I, ant1_Q)
            ant2_I, ant2_Q = ant_IQ_norm(ant2_I, ant2_Q)
            ant3_I, ant3_Q = ant_IQ_norm(ant3_I, ant3_Q)

            '''
            for i in range(len(ant0_I)):
                plt.plot([0,ant0_I[i]], [0, ant0_Q[i]], label='{}'.format(i))
                i = i+1
            plt.legend()
            plt.title('ant0 IQ vector after normalization')
            plt.show()
            '''
            #from the reference IQ vector pattern, there is a small change between every us except pi/2 caused by 250KHz frequency shift

            def calculate_angle(I1, Q1, I2, Q2):
                # 计算归一化后的点积
                dot_product = I1 * I2 + Q1 * Q2
                
                # 计算 cos(θ)
                cos_theta = np.clip(dot_product, -1.0, 1.0)
                
                # 计算角度 (弧度)
                theta = np.arccos(cos_theta)
                
                # 返回角度（可选：将弧度转为角度）
                angle_in_degrees = np.degrees(theta)
                
                return theta
        
            ref_theta_array = np.zeros(63)
            for i in range(63):
                ref_theta_array[i] = calculate_angle(reference_I[i], reference_Q[i], reference_I[i+1], reference_Q[i+1])
            # in every us, the IQ vector will rotate pi/2 + a small error angle
            angle_change_1us = np.mean(ref_theta_array)/0.125 - pi/2
            print('angle_change_1us:',angle_change_1us)
            # this function helps to compensate the 250KHz and a small error angle
            def rotate_vector(I, Q, theta):
                I_new = I * np.cos(theta) - Q * np.sin(theta)
                Q_new = I * np.sin(theta) + Q * np.cos(theta)
                
                return I_new, Q_new
            def compensate_phase(ant_I, ant_Q, angle_change_1us):
                for i in range(len(ant_I)):
                    rotate_angle = i*(3.14/2 + angle_change_1us)*8
                    ant_I[i], ant_Q[i] = rotate_vector(ant_I[i], ant_Q[i], -rotate_angle)
                return ant_I, ant_Q
            '''
            plt.figure()
            ax1 = plt.subplot(121)
            unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, label='Unit Circle')
            ax1.add_artist(unit_circle)
            for i in range(len(ant0_I)):
                ax1.plot([0,ant0_I[i]], [0, ant0_Q[i]], label='{}'.format(i))
                i = i+1
            ax1.set_title('ant0 IQ vector before compensation')
            ax1.set_xlim(-1,1)
            ax1.set_ylim(-1,1)
            plt.legend()
            '''
            ant0_I, ant0_Q = compensate_phase(ant0_I, ant0_Q, angle_change_1us)
            ant1_I, ant1_Q = compensate_phase(ant1_I, ant1_Q, angle_change_1us)
            ant2_I, ant2_Q = compensate_phase(ant2_I, ant2_Q, angle_change_1us)
            ant3_I, ant3_Q = compensate_phase(ant3_I, ant3_Q, angle_change_1us)
            '''
            ax2 = plt.subplot(122)
            unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, label='Unit Circle')
            ax2.add_artist(unit_circle)
            for i in range(len(ant0_I)):
                ax2.plot([0,ant1_I[i]], [0, ant1_Q[i]], label='{}'.format(i))
                i = i+1
            ax2.set_title('ant0 IQ vector after compensation')
            ax2.set_xlim(-1,1)
            ax2.set_ylim(-1,1)
            plt.legend()
            plt.show()
            '''
            
            ant0_I_mean, ant0_Q_mean = np.mean(ant0_I), np.mean(ant0_Q)
            ant1_I_mean, ant1_Q_mean = np.mean(ant1_I), np.mean(ant1_Q)
            ant2_I_mean, ant2_Q_mean = np.mean(ant2_I), np.mean(ant2_Q)
            ant3_I_mean, ant3_Q_mean = np.mean(ant3_I), np.mean(ant3_Q)

            rotate_angle_1_0 = 2*(np.pi/2 + angle_change_1us) #+ np.radians(24)
            rotate_angle_2_0 = 4*(np.pi/2 + angle_change_1us)
            rotate_angle_3_0 = 6*(np.pi/2 + angle_change_1us) #+ np.radians(24)

            ant1_I_mean, ant1_Q_mean = rotate_vector(ant1_I_mean, ant1_Q_mean, -rotate_angle_1_0)
            ant2_I_mean, ant2_Q_mean = rotate_vector(ant2_I_mean, ant2_Q_mean, -rotate_angle_2_0)
            ant3_I_mean, ant3_Q_mean = rotate_vector(ant3_I_mean, ant3_Q_mean, -rotate_angle_3_0)
            
            ant0_I_mean, ant0_Q_mean = normalization(ant0_I_mean, ant0_Q_mean)
            ant1_I_mean, ant1_Q_mean = normalization(ant1_I_mean, ant1_Q_mean)
            ant2_I_mean, ant2_Q_mean = normalization(ant2_I_mean, ant2_Q_mean)
            ant3_I_mean, ant3_Q_mean = normalization(ant1_I_mean, ant3_Q_mean)

            '''
            ax = plt.subplot(111)
            k = 0
            #ax.plot([0, np.min(ant0_I)], [0, ant0_Q[np.argmin(ant0_I)]], c = 'b', linewidth = 1, label = 'ant0_min')
            #ax.plot([0, np.max(ant0_I)], [0, ant0_Q[np.argmax(ant0_I)]], c = 'b', linewidth = 3, label = 'ant0_max')
            ax.plot([0, ant0_I_mean], [0, ant0_Q_mean], c = 'b', linestyle='--', label = 'ant0_mean')
            #ax.plot([0, ant0_I[k]], [0, ant0_Q[k]], c = 'b', linestyle='--', label = 'ant0_mean')
            #ax.plot([0, np.min(ant1_I)], [0, ant1_Q[np.argmin(ant1_I)]], c = 'y', linewidth = 1, label = 'ant1_min')
            #ax.plot([0, np.max(ant1_I)], [0, ant1_Q[np.argmax(ant1_I)]], c = 'y', linewidth = 3, label = 'ant1_max')
            ax.plot([0, ant1_I_mean], [0, ant1_Q_mean], c = 'y', linestyle='--', label = 'ant1_mean')
            #ax.plot([0, ant1_I[k]], [0, ant1_Q[k]], c = 'y', linestyle='--', label = 'ant1_mean')
            #ax.plot([0, np.min(ant2_I)], [0, ant2_Q[np.argmin(ant2_I)]], c = 'g', linewidth = 1, label = 'ant2_min')
            #ax.plot([0, np.max(ant2_I)], [0, ant2_Q[np.argmax(ant2_I)]], c = 'g', linewidth = 3, label = 'ant2_max')
            ax.plot([0, ant2_I_mean], [0, ant2_Q_mean], c = 'g', linestyle='--', label = 'ant2_mean')
            #ax.plot([0, ant2_I[k]], [0, ant2_Q[k]], c = 'g', linestyle='--', label = 'ant2_mean')
            ax.plot([0, ant3_I_mean], [0, ant3_Q_mean], c = 'pink', linestyle='--', label = 'ant3_mean')

            unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, label='Unit Circle')
            ax.add_artist(unit_circle)
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
            plt.legend()
            plt.show()
            '''

            #for i in range(len(ant0_I)):

            def steering_vector(alpha):
                j = 1j  # 复数单位
                return np.array([1, cmath.exp(-j * 2 * np.pi * 2.4e9 * (0.0375*np.sin(alpha)/SPEED_OF_LIGHT)), cmath.exp(-j * 2 * np.pi * 2.4e9 * 2*(0.0375*np.sin(alpha)/SPEED_OF_LIGHT))])

            def DoA_algorithm(ant0_I_mean, ant0_Q_mean, ant1_I_mean, ant1_Q_mean, ant2_I_mean, ant2_Q_mean):
                ant0_theta = cmath.phase(complex(ant0_I_mean, ant0_Q_mean))
                ant1_theta = cmath.phase(complex(ant1_I_mean, ant1_Q_mean))
                ant2_theta = cmath.phase(complex(ant2_I_mean, ant2_Q_mean))

                ant1_theta = ant1_theta - ant0_theta
                ant2_theta = ant2_theta - ant0_theta
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
            
            angle = DoA_algorithm(ant0_I_mean, ant0_Q_mean, ant1_I_mean, ant1_Q_mean, ant2_I_mean, ant2_Q_mean)
            print('DoA:', angle)

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

