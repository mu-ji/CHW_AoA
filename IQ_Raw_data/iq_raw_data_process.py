import numpy as np
import matplotlib.pyplot as plt
import AoA_cal_angle
import cmath
SPEED_OF_LIGHT  = 299792458



def cal_angle_list(true_angle):
    angle_list = []
    data = np.load('IQ_Raw_data/{}_data.npz'.format(true_angle))
    for i in range(len(data['rssi'])):
        I_data = data['I_data'][i]
        Q_data = data['Q_data'][i]

        reference_I = I_data[:8]
        reference_Q = Q_data[:8]

        ant0_I = I_data[9:88:8]
        ant0_Q = Q_data[9:88:8]

        ant1_I = I_data[11:88:8]
        ant1_Q = Q_data[11:88:8]

        ant2_I = I_data[13:88:8]
        ant2_Q = Q_data[13:88:8]

        ant3_I = I_data[15:88:8]
        ant3_Q = Q_data[15:88:8]


        def normalization(ant_I, ant_Q):
            # 计算幅度
            amplitude = np.sqrt(ant_I**2 + ant_Q**2)
            # 归一化 I 和 Q 分量
            norm_I = ant_I / amplitude
            norm_Q = ant_Q / amplitude
            return norm_I, norm_Q

        def ant_IQ_norm(ant_I_array, ant_Q_array):
            for i in range(len(ant_I_array)):
                ant_I_array[i], ant_Q_array[i] = normalization(ant_I_array[i], ant_Q_array[i])
            return ant_I_array, ant_Q_array
        
        # normalize three antenna IQ data, only consider phase information

        reference_I, reference_Q = ant_IQ_norm(reference_I, reference_Q)

        ant0_I, ant0_Q = ant_IQ_norm(ant0_I, ant0_Q)
        ant1_I, ant1_Q = ant_IQ_norm(ant1_I, ant1_Q)
        ant2_I, ant2_Q = ant_IQ_norm(ant2_I, ant2_Q)
        ant3_I, ant3_Q = ant_IQ_norm(ant3_I, ant3_Q)

        def calculate_angle(I1, Q1, I2, Q2):
            # 计算归一化后的点积
            dot_product = I1 * I2 + Q1 * Q2
            
            # 计算 cos(θ)
            cos_theta = np.clip(dot_product, -1.0, 1.0)
            
            # 计算角度 (弧度)
            theta = np.arccos(cos_theta)
            
            # 返回角度（可选：将弧度转为角度）
            angle_in_degrees = np.degrees(theta)
            
            return theta

        ref_theta_array = np.zeros(4)
        for i in range(4):
            ref_theta_array[i] = calculate_angle(reference_I[i], reference_Q[i], reference_I[i+4], reference_Q[i+4])
        
        # in every us, the IQ vector will rotate pi/2 + a small error angle
        angle_change_1us = np.mean(ref_theta_array)/4
        print('angle_change_1us:',angle_change_1us)
        # this function helps to compensate the 250KHz and a small error angle
        def rotate_vector(I, Q, theta):
            I_new = I * np.cos(theta) - Q * np.sin(theta)
            Q_new = I * np.sin(theta) + Q * np.cos(theta)
            
            return I_new, Q_new
        def compensate_phase(ant_I, ant_Q, angle_change_1us):
            for i in range(len(ant_I)):
                rotate_angle = i*(3.14/2 + angle_change_1us)*8
                ant_I[i], ant_Q[i] = rotate_vector(ant_I[i], ant_Q[i], -rotate_angle)
            return ant_I, ant_Q

        ant0_I_mean, ant0_Q_mean = np.mean(ant0_I), np.mean(ant0_Q)
        ant1_I_mean, ant1_Q_mean = np.mean(ant1_I), np.mean(ant1_Q)
        ant2_I_mean, ant2_Q_mean = np.mean(ant2_I), np.mean(ant2_Q)
        ant3_I_mean, ant3_Q_mean = np.mean(ant3_I), np.mean(ant3_Q)

        rotate_angle_1_0 = 2*(np.pi/2 + angle_change_1us) #+ np.radians(24)
        rotate_angle_2_0 = 4*(np.pi/2 + angle_change_1us)
        rotate_angle_3_0 = 6*(np.pi/2 + angle_change_1us) #+ np.radians(24)

        ant1_I_mean, ant1_Q_mean = rotate_vector(ant1_I_mean, ant1_Q_mean, -rotate_angle_1_0)
        ant2_I_mean, ant2_Q_mean = rotate_vector(ant2_I_mean, ant2_Q_mean, -rotate_angle_2_0)
        ant3_I_mean, ant3_Q_mean = rotate_vector(ant3_I_mean, ant3_Q_mean, -rotate_angle_3_0)
        
        ant0_I_mean, ant0_Q_mean = normalization(ant0_I_mean, ant0_Q_mean)
        ant1_I_mean, ant1_Q_mean = normalization(ant1_I_mean, ant1_Q_mean)
        ant2_I_mean, ant2_Q_mean = normalization(ant2_I_mean, ant2_Q_mean)
        ant3_I_mean, ant3_Q_mean = normalization(ant1_I_mean, ant3_Q_mean)

        def steering_vector(alpha):
            j = 1j  # 复数单位
            return np.array([1, cmath.exp(-j * 2 * np.pi * 2.4e9 * (0.0375*np.sin(alpha)/SPEED_OF_LIGHT)), cmath.exp(-j * 2 * np.pi * 2.4e9 * 2*(0.0375*np.sin(alpha)/SPEED_OF_LIGHT))])

        def DoA_algorithm(ant0_I_mean, ant0_Q_mean, ant1_I_mean, ant1_Q_mean, ant2_I_mean, ant2_Q_mean):
            ant0_theta = cmath.phase(complex(ant0_I_mean, ant0_Q_mean))
            ant1_theta = cmath.phase(complex(ant1_I_mean, ant1_Q_mean))
            ant2_theta = cmath.phase(complex(ant2_I_mean, ant2_Q_mean))

            ant1_theta = ant1_theta - ant0_theta
            ant2_theta = ant2_theta - ant0_theta
            ant0_theta = 0

            received_signal = np.array([cmath.exp(1j*ant0_theta), cmath.exp(1j*ant1_theta), cmath.exp(1j*ant2_theta)])
            angle_list = [np.radians(i) for i in range(-90, 90)]
            y_alpha_list = []
            for alpha in angle_list:
                y_alpha = steering_vector(alpha)[0]*received_signal[0] + steering_vector(alpha)[1]*received_signal[1] + steering_vector(alpha)[2]*received_signal[2]
                y_alpha_list.append(y_alpha)

            #plt.plot([i for i in range(-90, 90)], y_alpha_list)
            #plt.show()
            return [i for i in range(-90, 90)][np.argmax(np.array(y_alpha_list))]
        
        angle = DoA_algorithm(ant0_I_mean, ant0_Q_mean, ant1_I_mean, ant1_Q_mean, ant2_I_mean, ant2_Q_mean)
        #print('DoA:', angle)
        angle_list.append(angle)
    return angle_list

true_angle_list = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30 , 40, 50, 60]

angle_list_array = []
for true_angle in true_angle_list:
    angle_list_array.append(np.array(cal_angle_list(true_angle)))

plt.figure()
#plt.plot([i for i in range(200)], angle_list)
plt.violinplot(np.array(angle_list_array).T,showmeans=True)
#plt.plot(true_angle_list,true_angle_list)
plt.grid()
plt.show()
