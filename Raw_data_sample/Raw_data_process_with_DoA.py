import numpy as np
import matplotlib.pyplot as plt
import AoA_cal_angle

import cmath

SPEED_OF_LIGHT  = 299792458

data = np.load('Raw_data_sample/0006_2.5_(0,0)_data.npz')

#phase_data shape 576*200 200 measurements and 576 samples for each measurement

data_list = [0, 10, 20, 30, 40, 50]

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

for x_index in data_list:
    data = np.load('Raw_data_sample/0006_2.5_({},0)_data.npz'.format(x_index))

    for packet_id in range(200):

        phase_data = data['phase_data'][packet_id]
        reference_ant_data = np.zeros_like(phase_data)
        reference_ant_data[0:64] = phase_data[0:64]
        reference_ant_data,reference_slope = AoA_cal_angle.complete_reference_phase_data(reference_ant_data)

        i = 72
        antenna0_phase_data_1 = data['phase_data'][packet_id][72:80]
        antenna0_phase_data_2 = data['phase_data'][packet_id][328:336]

        antenna1_phase_data_1 = data['phase_data'][packet_id][88:96] - 2*reference_slope
        antenna1_phase_data_2 = data['phase_data'][packet_id][344:352]

        antenna2_phase_data_1 = data['phase_data'][packet_id][104:112] - 4*reference_slope
        antenna2_phase_data_2 = data['phase_data'][packet_id][360:368]

        angle_list = []
        for sample_id in range(8):
            ant0_I = np.cos(antenna0_phase_data_1[sample_id])
            ant0_Q = np.sin(antenna0_phase_data_1[sample_id])

            ant1_I = np.cos(antenna1_phase_data_1[sample_id])
            ant1_Q = np.sin(antenna1_phase_data_1[sample_id])

            ant2_I = np.cos(antenna2_phase_data_1[sample_id])
            ant2_Q = np.sin(antenna2_phase_data_1[sample_id])

            angle = DoA_algorithm(ant0_I, ant0_Q, ant1_I, ant1_Q, ant2_I, ant2_Q)
            angle_list.append(angle)
            
        print(np.mean(np.array(angle_list)))







