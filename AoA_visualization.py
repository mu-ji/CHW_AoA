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

def angle_cal_from_serial():
    ser = serial.Serial('COM11', 115200)

    rawFrame = []

    #while iteration < times:
    while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-3:]==[255, 255, 255]:
            if len(rawFrame) == 1288:
                received_data = rawFrame[:1280]
                num_samples = 320

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


                reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)
                ant1_data = AoA_cal_angle.complete_other_phase_data(ant1_data,reference_slope)
                ant2_data = AoA_cal_angle.complete_other_phase_data(ant2_data,reference_slope)
                ant3_data = AoA_cal_angle.complete_other_phase_data(ant3_data,reference_slope)

                diff1 = ant1_data[72:80] - ant3_data[72:80]
                diff2 = ant2_data[88:96] - ant3_data[88:96]
                
                mean_phase_diff_1 = np.mean(AoA_cal_angle.release_jump(diff1))
                mean_phase_diff_2 = np.mean(AoA_cal_angle.release_jump(diff2))

                wave_length = 0.125 # meter  maybe need add the frequency offset?
                antenna_interval = 0.0375 # m

                angle1 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_1,wave_length,antenna_interval)
                angle2 = AoA_cal_angle.two_ant_cal_angle(mean_phase_diff_2,wave_length,2*antenna_interval)

                if np.std((angle1,angle2)) >= 20:
                    #print('not stable')
                    return -180
                else:
                    #print('estimate angle:', np.mean((angle1,angle2)))
                    return np.mean((angle1,angle2)) - 90
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
    angle = angle_cal_from_serial()
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