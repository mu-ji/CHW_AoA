import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii

import matplotlib.pyplot as plt
from math import pi, atan2, sqrt
from scipy.linalg import eig

ser = serial.Serial('COM11', 115200)

SPEED_OF_LIGHT  = 299792458
frequency = 16000000
num_iterations = 200     # 进行的循环次数
iteration = 0

rawFrame = []

all_data = {
    'phase_data': [],
    'mag_data': [],
    'rssi' : []
}


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

            all_data['phase_data'].append(phase_data)
            all_data['mag_data'].append(mag_data)


            rssi = bytes(rawFrame[-8:-4])
            rssi = int(rssi.decode('utf-8'))
            #print(iteration)
            all_data['rssi'].append(rssi)
            print(iteration, rssi)
            #print('packet_number:',rawFrame[-4])
            #print('-------------------------------')

            
        rawFrame = []
        iteration = iteration + 1

    if len(all_data['phase_data']) == num_iterations:
        all_data['phase_data'] = np.array(all_data['phase_data'])
        all_data['mag_data'] = np.array(all_data['mag_data'])
        all_data['rssi'] = np.array(all_data['rssi'])

        np.savez('Raw_data_sample/0006_2_(10,0)_data.npz', **all_data)
        break

