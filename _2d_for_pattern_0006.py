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
    ser = serial.Serial('COM8', 115200)
    rawFrame = []
    #while iteration < times:
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
                    phase_data[i] = phase[0]
                    mag_data[i] = mag[0]
                phase_data = phase_data.astype(np.float32)
                mag_data = mag_data.astype(np.float32)

                antenna_phase_array = np.zeros((4,4,8))

                reference_ant_data = np.zeros_like(phase_data)
                antenna_phase_array = np.zeros((4,4,len(phase_data)))
                reference_ant_data[0:64] = phase_data[0:64]
                reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)

                antenna_data_list = []
                i = 72
                while i < 576:
                    temp_antenna_data = np.zeros_like(phase_data)
                    temp_antenna_data[i:i+8] = phase_data[i:i+8]
                    temp_antenna_data = AoA_cal_angle.complete_other_phase_data(temp_antenna_data, reference_slope)
                    antenna_data_list.append(temp_antenna_data)
                    i = i + 16

                first_switch_data = np.array(antenna_data_list)[:16,:]
                second_switch_data = np.array(antenna_data_list)[16:,:]

                def cal_angle(antenna_data):
                    diff1 = antenna_data[1,:] - antenna_data[0,:]
                    diff1 = AoA_cal_angle.release_jump(diff1)
                    diff2 = antenna_data[2,:] - antenna_data[1,:]
                    diff2 = AoA_cal_angle.release_jump(diff2)

                    wave_length = 0.125 # meter  maybe need add the frequency offset?
                    antenna_interval = 0.0375 # m

                    angle1 = AoA_cal_angle.two_ant_cal_angle(np.mean(diff1),wave_length,antenna_interval)
                    angle2 = AoA_cal_angle.two_ant_cal_angle(np.mean(diff2),wave_length,antenna_interval)
                    return angle1, angle2
                
                first_x_angle1_1, first_x_angle1_2 = cal_angle(first_switch_data[0:3,:])
                first_x_angle2_1, first_x_angle2_2 = cal_angle(first_switch_data[3:6,:])

                second_x_angle1_1, second_x_angle1_2 = cal_angle(second_switch_data[0:3,:])
                second_x_angle2_1, second_x_angle2_2 = cal_angle(second_switch_data[3:6,:])

                first_y_angle1_1, first_y_angle1_2 = cal_angle(first_switch_data[6:9,:])
                first_y_angle2_1, first_y_angle2_2 = cal_angle(first_switch_data[9:12,:])

                second_y_angle1_1, second_y_angle1_2 = cal_angle(second_switch_data[6:9,:])
                second_y_angle2_1, second_y_angle2_2 = cal_angle(second_switch_data[9:12,:])

                x_angle_array = [first_x_angle1_1, first_x_angle1_2, first_x_angle2_1, first_x_angle2_2, second_x_angle1_1, second_x_angle1_2, second_x_angle2_1, second_x_angle2_2]
                y_angle_array = [first_y_angle1_1, first_y_angle1_2, first_y_angle2_1, first_y_angle2_2, second_y_angle1_1, second_y_angle1_2, second_y_angle2_1, second_y_angle2_2]

                return x_angle_array,y_angle_array
            rawFrame = []

# 设置图像大小和属性
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2,2,1)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.set_title('Angle of Arrival x_angle estimation')
circle = plt.Circle((0, 0), 1, fill=False, color='grey', linewidth=2)
ax1.add_artist(circle)
ax1.set_position([0.05, 0.65, 0.2, 0.2])

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
ax2.set_position([0.05, 0.25, 0.2, 0.2])

# 绘制指示角度的线和文本
line2, = ax2.plot([], [], 'b-', lw=2, label = 'estimated y_angle')
text2 = ax2.text(0, 0, '', ha='center', va='bottom')

ax3 = fig.add_subplot(2,2,3)
ax3.set_xlim(-1, 1)
ax3.set_ylim(-1, 1)
ax3.set_aspect('equal')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.scatter([0],[0],label = 'True Position')
ax3.set_title('2D Angle of Arrival Visualization')
x_ticks = np.linspace(-0.5, 0.5, 21)
y_ticks = np.linspace(-0.5, 0.5, 21)
ax3.set_xticks(x_ticks)
ax3.set_yticks(y_ticks)
ax3.grid()
ax3.set_position([0.2, 0.1, 0.8, 0.8])


# 绘制指示角度的线和文本
dot, = ax3.plot([], [], 'ro', markersize=10, label = 'Estimate Position')

# 定义动画更新函数
plt.legend()

def update_earlier_measurement(data_array, new_data):

    if data_array.shape[0] < 10:
        # data_array未满,直接添加新数据
        data_array = np.vstack((data_array, new_data))
    else:
        # data_array已满,删除最早的数据并添加新数据
        data_array = np.delete(data_array, 0, axis=0)
        data_array = np.vstack((data_array, new_data))
    
    return data_array


x_earlier_measurement = np.zeros((0,8))
y_earlier_measurement = np.zeros((0,8))

def animate(frame):
    global x_earlier_measurement
    global y_earlier_measurement
    anchor_x = 0
    anchor_y = 0
    height = 2.6
    x_angle_array,y_angle_array = array_angle_cal_from_serial()

    x_earlier_measurement = update_earlier_measurement(x_earlier_measurement, x_angle_array)
    y_earlier_measurement = update_earlier_measurement(y_earlier_measurement, y_angle_array)

    x_angle = AoA_filter.mean_filter(np.mean(x_earlier_measurement, axis=0))
    y_angle = AoA_filter.mean_filter(np.mean(y_earlier_measurement, axis=0))

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

    
    x_dot = height/np.tan(np.radians(x_angle)) + anchor_x
    y_dot = -height/np.tan(np.radians(y_angle)) + anchor_y
    #print('x_angle:', x_angle, x_dot)
    #print('y_angle:', y_angle, y_dot)
    dot.set_data(x_dot, y_dot)

    return [dot, line1, text1, line2, text2]

# 创建并运行动画
ani = FuncAnimation(fig, animate, frames=np.linspace(0, 360, 360), interval=50, blit=True)

# 显示动画
plt.show()