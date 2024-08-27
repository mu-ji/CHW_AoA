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
        print(1)
        print(len(rawFrame))
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
            
            #phase_data_diff = np.diff(phase_data)

            iteration = iteration + 1
            

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([i*0.125 for i in range(num_samples)], phase_data, marker='*')
            ax.plot([8*8*0.125]*2, [-201, 201], c='b')
            i = 8
            while i < 72:
                switch_slot = plt.Rectangle(xy=(8*i*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'r')
                ax.add_patch(switch_slot)
                sample_slot = plt.Rectangle(xy=(8*(i+1)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'g')
                ax.add_patch(sample_slot)
                i = i + 2

            plt.legend()
            plt.show()

            antenna_id = 0
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([8*8*0.125]*2, [-201, 201], c='b')
            ax.plot([i*0.125 for i in range(num_samples)], phase_data, marker='*')
            switch_slot_1 = plt.Rectangle(xy=(8*(8+antenna_id)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'r')
            ax.add_patch(switch_slot_1)
            sample_slot_1 = plt.Rectangle(xy=(8*(8+antenna_id+1)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'g')
            ax.add_patch(sample_slot_1)
            switch_slot_2 = plt.Rectangle(xy=(8*(8+antenna_id+32)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'r')
            ax.add_patch(switch_slot_2)
            sample_slot_2 = plt.Rectangle(xy=(8*(8+antenna_id+1+32)*0.125 , -201), width=1, height=402, alpha=0.2, angle=0.0, color = 'g')
            ax.add_patch(sample_slot_2)
            plt.legend()
            plt.show()

            antenna_id = 15
            fig, ax = plt.subplots(figsize=(10, 6))
            sample_1 = phase_data[64+(antenna_id+1)*8:64+(antenna_id+1)*8 + 8]
            sample_2 = phase_data[64+(antenna_id+1)*8 + 256:64+(antenna_id+1)*8 + 8 + 256]
            ax.plot([i * 0.125 for i in range(64+(antenna_id+1)*8, 64+(antenna_id+1)*8 + 8)], sample_1, label='sample 1')
            ax.plot([i * 0.125 for i in range(64+(antenna_id+1)*8 + 256, 64+(antenna_id+1)*8 + 8 + 256)], sample_2, label='sample 2')
            ax.plot([i * 0.125 for i in range(64+(antenna_id+1)*8, 64+(antenna_id+1)*8 + 8)], sample_2, label='move sample 2 to sample 1')

            plt.legend()
            plt.show()

            
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

