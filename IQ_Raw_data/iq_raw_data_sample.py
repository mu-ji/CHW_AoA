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
ser = serial.Serial('COM8', 115200)

import cmath

num_iterations = 200     # 进行的循环次数
iteration = 0

rawFrame = []

all_data = {
    'I_data': [],
    'Q_data': [],
    'rssi' : [],
    'pattern' : []
}


num_samples = 88
while True:
    byte  = ser.read(1)        
    rawFrame += byte
    #print(len(rawFrame))
    if rawFrame[-3:]==[255, 255, 255]:
        if len(rawFrame) == 4*num_samples+8:
            received_data = rawFrame[:4*num_samples]
            num_samples = 88

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

            all_data['I_data'].append(I_data)
            all_data['Q_data'].append(Q_data)


            rssi = bytes(rawFrame[-8:-4])
            rssi = int(rssi.decode('utf-8'))
            #print(iteration)
            all_data['rssi'].append(rssi)
            print(iteration, rssi, len(all_data['rssi']))
            #print('packet_number:',rawFrame[-4])
            #print('-------------------------------')

            
        rawFrame = []
        iteration = iteration + 1
    if len(all_data['I_data']) == num_iterations:
        all_data['I_data'] = np.array(all_data['I_data'])
        all_data['Q_data'] = np.array(all_data['Q_data'])
        all_data['rssi'] = np.array(all_data['rssi'])
        all_data['pattern'] = ['42,43,44,41']

        np.savez('IQ_Raw_data/60_data.npz', **all_data)
        break
