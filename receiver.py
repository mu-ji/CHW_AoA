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
            
            #phase_data_diff = np.diff(phase_data)

            iteration = iteration + 1
            '''
            #fig = plt.figure()
            #fig.add_subplot(121)
            #plt.plot([i for i in range(160)],phase)
            #fig.add_subplot(122)
            #plt.plot([i for i in range(160)],phase_2pi)
            #plt.show()

            fig = plt.figure()
            plt.plot([i*0.125 for i in range(160)],phase_data, marker='*')
            plt.plot([i*0.125 for i in range(160)],mag_data, marker='*')

            i = 8
            plt.plot([8*i*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+1)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+4)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+5)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+8)*0.125]*2, [-201,201],c = 'b')
            plt.plot([8*(i+9)*0.125]*2, [-201,201],c = 'b')

            i = 8+2
            plt.plot([8*i*0.125]*2, [-201,201],c = 'g')
            plt.plot([8*(i+1)*0.125]*2, [-201,201],c = 'g')
            plt.plot([8*(i+4)*0.125]*2, [-201,201],c = 'g')
            plt.plot([8*(i+5)*0.125]*2, [-201,201],c = 'g')
            plt.plot([8*(i+8)*0.125]*2, [-201,201],c = 'g')
            plt.plot([8*(i+9)*0.125]*2, [-201,201],c = 'g')


            plt.plot([160*0.125]*2,[-201,201],c='r')
            plt.xlabel('us')
            plt.ylabel('phase')
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

