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

from matplotlib.animation import FuncAnimation

def array_angle_cal_from_serial():
    ser = serial.Serial('COM11', 115200)
    rawFrame = []
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
                    phase_data[i] = phase[0]
                    mag_data[i] = mag[0]
                phase_data = phase_data.astype(np.float32)
                mag_data = mag_data.astype(np.float32)

                antenna_phase_array = np.zeros((4,4,8))

                reference_ant_data = np.zeros_like(phase_data)
                antenna_phase_array = np.zeros((4,4,len(phase_data)))
                reference_ant_data[0:64] = phase_data[0:64]
                reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)
                '''
                plt.figure()
                i = 8
                plt.plot([i*0.125 for i in range(320)],phase_data, marker='*')
                plt.plot([i*0.125 for i in range(320)],mag_data, marker='*')
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
                '''
                k = 72
                for i in range(4):
                    for j in range(4):
                        #if i == 0 and j == 0:
                        #    antenna_phase_array[i][j] = AoA_cal_angle.complete_other_phase_data(phase_data[312:320],reference_slope)
                        temple_phase = np.zeros_like(phase_data)
                        temple_phase[k:k+8] = phase_data[k:k+8]
                        antenna_phase_array[i][j] = AoA_cal_angle.complete_other_phase_data(temple_phase,reference_slope)
                        #antenna_phase_array[i][j] = phase_data[k:k+8]
                        k = k + 16

                wave_length = 0.125 # m
                antenna_interval = 0.0375 # m
                x_angle_array = AoA_cal_angle.array_ant_cal_x_angle(antenna_phase_array, wave_length, antenna_interval,)
                y_angle = 0

                wave_length = 0.125 # m
                antenna_interval = 0.0375 # m


                #print('angle1:',angle1)
                #print('angle2:',angle2)
                return x_angle_array,y_angle
            rawFrame = []

# 设置图像大小和属性
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Angle of Arrival Visualization')

# 绘制单位圆
circle = plt.Circle((0, 0), 1, fill=False, color='grey', linewidth=2)
ax.add_artist(circle)

# 绘制指示角度的线和文本
line1, = ax.plot([], [], 'r-', lw=2, label = 'row1 estimated angle')
text1 = ax.text(0, 0, '', ha='center', va='bottom')

line2, = ax.plot([], [], 'b-', lw=2, label = 'row2 estimated angle')
text2 = ax.text(0, 0, '', ha='center', va='bottom')

line3, = ax.plot([], [], 'g-', lw=2, label = 'row3 estimated angle')
text3 = ax.text(0, 0, '', ha='center', va='bottom')

line4, = ax.plot([], [], 'y-', lw=2, label = 'row4 estimated angle')
text4 = ax.text(0, 0, '', ha='center', va='bottom')
# 定义动画更新函数
plt.legend()
def animate(frame):
    x_angle_array,y_angle = array_angle_cal_from_serial()
    x1 = np.cos(np.radians(x_angle_array[0]))
    y1 = np.sin(np.radians(x_angle_array[0]))
    line1.set_data([0, x1], [0, y1])
    text1.set_position((0.9 * x1, 0.9 * y1))
    text1.set_text(f'{x_angle_array[0]:.2f}°')

    x2 = np.cos(np.radians(x_angle_array[1]))
    y2 = np.sin(np.radians(x_angle_array[1]))
    line2.set_data([0, x2], [0, y2])
    text2.set_position((0.9 * x2, 0.9 * y2))
    text2.set_text(f'{x_angle_array[1]:.2f}°')

    x3 = np.cos(np.radians(x_angle_array[2]))
    y3 = np.sin(np.radians(x_angle_array[2]))
    line3.set_data([0, x3], [0, y3])
    text3.set_position((0.9 * x3, 0.9 * y3))
    text3.set_text(f'{x_angle_array[2]:.2f}°')

    x4 = np.cos(np.radians(x_angle_array[3]))
    y4 = np.sin(np.radians(x_angle_array[3]))
    line4.set_data([0, x4], [0, y4])
    text4.set_position((0.9 * x1, 0.9 * y1))
    text4.set_text(f'{x_angle_array[3]:.2f}°')
    return [line1, text1, line2, text2, line3, text3, line4, text4]

# 创建并运行动画
ani = FuncAnimation(fig, animate, frames=np.linspace(0, 360, 360), interval=50, blit=True)

# 显示动画
plt.show()