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

def AoA_cal():
    ser = serial.Serial('COM11', 115200)

    rawFrame = []

    diff_list = []

    #while iteration < times:
    while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-3:]==[255, 255, 255]:
            if len(rawFrame) == 648:
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
                
                #phase_data_diff = np.diff(phase_data)

                reference_ant_data = np.zeros_like(phase_data)
                ant1_data = np.zeros_like(phase_data)
                ant2_data = np.zeros_like(phase_data)
                ant3_data = np.zeros_like(phase_data)

                reference_ant_data[0:64] = phase_data[0:64]
                ant1_data[72:80] = phase_data[72:80]
                ant2_data[88:96] = phase_data[88:96]
                ant3_data[104:112] = phase_data[104:112]

                def complete_reference_phase_data(reference_ant_data):
                    diff = reference_ant_data[1:]-reference_ant_data[:-1]
                    for i in range(len(diff)):
                        if abs(diff[i]) >= 100:
                            diff[i] = 0
                    slope = np.mean(diff[diff != 0])
                    for i in range(len(reference_ant_data)):
                        if reference_ant_data[i] != 0:
                            continue
                        else:
                            next_phase = reference_ant_data[i-1] + slope
                            if next_phase > 201:
                                next_phase = next_phase - 402
                            reference_ant_data[i] = next_phase
                    return reference_ant_data, slope/0.125

                def find_best_intercept(x, y, slope):
                    n = len(x)
                    A = np.vstack([x, np.ones(n)]).T
                    b = np.sum(y - np.multiply(x,slope)) / n
                    return b
                    
                def complete_other_phase_data(ant_phase_data,reference_slope):
                    nonzero_indices = np.nonzero(ant_phase_data)[0]
                    min_idx = nonzero_indices[0]
                    max_idx = nonzero_indices[-1]
                    x = [i*0.125 for i in range(min_idx,max_idx)]
                    y = ant_phase_data[min_idx:max_idx]
                    for i in range(len(y)-1):
                        if y[i]-y[i+1] >= 200:
                            y[i+1] = y[i+1] + 402

                    intercept = find_best_intercept(x,y,reference_slope)
                    first_index = min_idx
                    last_index = min_idx + 1
                    ant_phase_data[first_index] = reference_slope*first_index*0.125 + intercept
                    ant_phase_data[last_index] = reference_slope*last_index*0.125 + intercept
                    while first_index != 0 or last_index != (len(ant_phase_data)-1):
                        if first_index != 0:
                            last_phase = ant_phase_data[first_index] - 0.125*reference_slope
                            if last_phase <= -201:
                                last_phase = last_phase + 402
                            ant_phase_data[first_index-1] = last_phase
                            first_index = first_index - 1
                        if last_index != (len(ant_phase_data)-1):
                            next_phase = ant_phase_data[last_index] + 0.125*reference_slope
                            if next_phase > 201:
                                next_phase = next_phase - 402
                            ant_phase_data[last_index+1] = next_phase
                            last_index = last_index + 1
                    
                    return ant_phase_data

                reference_ant_data,reference_slope = complete_reference_phase_data(reference_ant_data)
                ant1_data = complete_other_phase_data(ant1_data,reference_slope)
                ant2_data = complete_other_phase_data(ant2_data,reference_slope)
                ant3_data = complete_other_phase_data(ant3_data,reference_slope)

                diff1 = ant1_data[72:80] - ant3_data[72:80]
                diff2 = ant2_data[88:96] - ant3_data[88:96]

                def release_jump(diff):
                    for i in range(len(diff)):
                        if abs(diff[i]) > 201:
                            if diff[i] < -201:
                                diff[i] = diff[i] + 402
                            if diff[i] > 201:
                                diff[i] = diff[i] - 402
                    return diff
                
                mean_phase_diff_1 = np.mean(release_jump(diff1))
                mean_phase_diff_2 = np.mean(release_jump(diff2))

                wave_length = 0.125 # m
                antenna_interval = 0.0375 # m

                angle1 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_1,wave_length,antenna_interval)
                angle2 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_2,wave_length,2*antenna_interval)

                if np.std((angle1,angle2)) >= 20:
                    #print('not stable')
                    return 0
                else:
                    #print('estimate angle:', np.mean((angle1,angle2)))
                    return np.mean((angle1,angle2))
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
line, = ax.plot([], [], 'r-', lw=2)
text = ax.text(0, 0, '', ha='center', va='bottom')

# 定义动画更新函数
def animate(frame):
    angle = AoA_cal()
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    line.set_data([0, x], [0, y])
    text.set_position((0.9 * x, 0.9 * y))
    text.set_text(f'{angle:.2f}°')
    return [line, text]

# 创建并运行动画
ani = FuncAnimation(fig, animate, frames=np.linspace(0, 360, 360), interval=50, blit=True)

# 显示动画
plt.show()