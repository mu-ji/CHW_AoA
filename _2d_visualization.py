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
import AoA_filter

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
                x_angle_array = AoA_cal_angle.array_ant_cal_x_angle_array(antenna_phase_array, wave_length, antenna_interval)
                y_angle_array = AoA_cal_angle.array_ant_cal_y_angle_array(antenna_phase_array, wave_length, antenna_interval)
                y_angle = 0

                wave_length = 0.125 # m
                antenna_interval = 0.0375 # m


                #print('angle1:',angle1)
                #print('angle2:',angle2)
                return x_angle_array,y_angle_array
            rawFrame = []

# 设置图像大小和属性
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.set_title('Angle of Arrival x_angle estimation')
circle = plt.Circle((0, 0), 1, fill=False, color='grey', linewidth=2)
ax1.add_artist(circle)

# 绘制指示角度的线和文本
line1, = ax1.plot([], [], 'r-', lw=2, label = 'estimated x_angle')
text1 = ax1.text(0, 0, '', ha='center', va='bottom')

ax2 = fig.add_subplot(2,2,2)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.set_aspect('equal')
ax2.set_title('Angle of Arrival y_angle estimation')
circle = plt.Circle((0, 0), 1, fill=False, color='grey', linewidth=2)
ax2.add_artist(circle)

# 绘制指示角度的线和文本
line2, = ax2.plot([], [], 'b-', lw=2, label = 'estimated y_angle')
text2 = ax2.text(0, 0, '', ha='center', va='bottom')

ax3 = fig.add_subplot(2,2,3)
ax3.set_xlim(-10, 10)
ax3.set_ylim(-10, 10)
ax3.set_aspect('equal')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('2D Angle of Arrival Visualization')


# 绘制指示角度的线和文本
dot, = ax3.plot([], [], 'ro', markersize=10)

# 定义动画更新函数
plt.legend()
def animate(frame):
    anchor_x = 0
    anchor_y = 0
    height = 1
    x_angle_array,y_angle_array = array_angle_cal_from_serial()
    x_angle = AoA_filter.mean_filter(x_angle_array)
    y_angle = AoA_filter.mean_filter(y_angle_array)

    x1 = np.cos(np.radians(x_angle))
    y1 = np.sin(np.radians(x_angle))
    line1.set_data([0, x1], [0, y1])
    text1.set_position((0.9 * x1, 0.9 * y1))
    text1.set_text(f'{x_angle:.2f}°')

    x2 = np.cos(np.radians(y_angle))
    y2 = np.sin(np.radians(y_angle))
    line2.set_data([0, x2], [0, y2])
    text2.set_position((0.9 * x2, 0.9 * y2))
    text2.set_text(f'{y_angle:.2f}°')

    
    x_dot = height/np.tan(x_angle) + anchor_x
    y_dot = height/np.tan(y_angle) + anchor_y
    print(x_dot, y_dot)
    dot.set_data(x_dot, y_dot)

    return [dot, line1, text1, line2, text2]

# 创建并运行动画
ani = FuncAnimation(fig, animate, frames=np.linspace(0, 360, 360), interval=50, blit=True)

# 显示动画
plt.show()