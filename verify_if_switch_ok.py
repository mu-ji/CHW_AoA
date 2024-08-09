
import serial
import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import binascii
import threading
import matplotlib.pyplot as plt
from math import pi, atan2, sqrt
from scipy.linalg import eig
import time

import AoA_algorithm
import AoA_cal_angle

ser_351 = serial.Serial('COM11', 115200)
ser_035 = serial.Serial('COM9', 115200)

SPEED_OF_LIGHT  = 299792458
frequency = 16000000

rawFrame_351 = []
rawFrame_035 = []

diff_list = []

times = 10
iteration = 0
#while iteration < times:
def receiver(ser,rawFrame):
    while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-3:]==[255, 255, 255]:
            if len(rawFrame) == 648:
                packet_number = np.expand_dims(rawFrame[-4], axis=0)
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
                
                phase_data = np.concatenate([phase_data, packet_number], axis=0)
                #phase_data_diff = np.diff(phase_data)
                if ser.port == 'COM11':
                    np.save('phase_data_351.npy',phase_data)
                if ser.port == 'COM9':
                    np.save('phase_data_035.npy',phase_data)

                
                try:
                    response_rssi = bytes(rawFrame[-8:-4])
                    response_rssi = int(response_rssi.decode('utf-8'))
                    #print(iteration)
                    print(ser.port+'{}'.format(response_rssi))
                    #print('packet_number:',rawFrame[-4])
                    #print('-------------------------------')

                except:
                    rawFrame = []
                    continue
                
            rawFrame = []



thread_035 = threading.Thread(target=receiver, args = (ser_035,rawFrame_035))
thread_351 = threading.Thread(target=receiver, args = (ser_351,rawFrame_351))

thread_351.start()
thread_035.start()

time.sleep(1)
phase_351 = np.load('phase_data_351.npy')
phase_035 = np.load('phase_data_035.npy')

phase_351_packet_number = phase_351[-1]
phase_035_packet_number = phase_035[-1]

if phase_035_packet_number == phase_351_packet_number:

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
    plt.plot([i*0.125 for i in range(160)],phase_351[:-1],label = 'id 351', marker='*')
    plt.plot([i*0.125 for i in range(160)],phase_035[:-1],label = 'id 035', marker='*')
    plt.legend()
    plt.show()
else:
    print('not the same packet')