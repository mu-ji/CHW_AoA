
import numpy as np
import matplotlib.pyplot as plt
import AoA_cal_angle
import random
import scipy.linalg as LA
import AoA_filter

class partical_filter:
    def __init__(self, partical_number):
        self.partical_number = partical_number
        self.x = 90
        self.partical_list = [random.uniform(0, 180) for _ in range(self.partical_number)]
        self.partical_weight = [0]*self.partical_number
        self.error = 0

    def update_weight(self, measurement):
        for i in range(self.partical_number):

            self.partical_weight[i] = 1/np.sum(np.abs(np.array([self.partical_list[i]]*8) - np.array(measurement)))
        self.partical_weight = [x / np.sum(self.partical_weight) for x in self.partical_weight]
    
    def update_x(self):
        self.x = np.sum(x * y for x, y in zip(self.partical_list, self.partical_weight))
        weight_std = np.std(self.partical_weight)

        if self.error > 10:
            weight_std = 2*1e-3
        print(weight_std)
        self.partical_list = [random.uniform(self.x - 5*weight_std*1e3, self.x + 5*weight_std*1e3) for _ in range(self.partical_number)]

    def update_error(self, measurement):
        self.error = np.mean(np.abs(np.array([self.x]*8) - np.array(measurement)))

packet_number = 0

true_x_angle = 10
0
data = np.load('Raw_data_sample/0006_2.5_({},0)_data.npz'.format(true_x_angle))

PF = partical_filter(partical_number=100)
KF = AoA_filter.Kalman_Filter()
x_earlier_measurement = np.zeros((0,8))

def update_earlier_measurement(data_array, new_data):

    if data_array.shape[0] < 10:
        # data_array未满,直接添加新数据
        data_array = np.vstack((data_array, new_data))
    else:
        # data_array已满,删除最早的数据并添加新数据
        data_array = np.delete(data_array, 0, axis=0)
        data_array = np.vstack((data_array, new_data))
    
    return data_array

PF_x_angle_list = []
KF_x_angle_list = []
while packet_number < 200:
    phase_data = data['phase_data'][packet_number]
    mag_data = data['mag_data'][packet_number]
    rssi = data['rssi'][packet_number]

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
        
        mag = np.mean(mag_data[i:i+8])
        complex_signal = [mag]*len(temp_antenna_data) * np.exp(1j * temp_antenna_data)

        #antenna_data_list.append(complex_signal)
        antenna_data_list.append(temp_antenna_data)
        i = i + 16

    first_switch_data = np.array(antenna_data_list)[:16,:]
    second_switch_data = np.array(antenna_data_list)[16:,:]
    '''
    def MUSIC(X):
        d = np.array([0, 0.0375, 0.075])
        n = 576
        Rxx = X@(X.conj().T)/n
        D,EV = LA.eig(Rxx)
        index = np.argsort(D)
        EN = EV.T[index].T[:,0:3-1]

        Angles = np.linspace(0, np.pi, 360)
        numAngles = Angles.size
        for i in range(numAngles):
            a = np.exp(-1j*2*np.pi*d.reshape(-1,1)*np.sin(Angles[i]))
            SP[i] = ((a.conj().T@a)/(a.conj().T@EN@EN.conj().T@a))[0,0]
    
        return SP
    
    X = np.array([antenna_data_list[0],
                    antenna_data_list[1],
                    antenna_data_list[2]])

    #Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    Angles = np.linspace(0, np.pi, 360)
    numAngles = Angles.size
    SP = np.empty(numAngles,dtype=complex)
    SP = MUSIC(X)

    SP = np.abs(SP)
    SPmax = np.max(SP)
    SP = 10 * np.log10(SP / SPmax)
    x = Angles * 180 / np.pi
    plt.plot(x, SP)
    plt.show()
    '''
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

    PF.update_error(x_angle_array)
    PF.update_weight(x_angle_array)
    PF.update_x()

    x_earlier_measurement = update_earlier_measurement(x_earlier_measurement, x_angle_array)

    KF.update_R(x_earlier_measurement)
    KF.predict()
    KF.update(x_angle_array)
    x_angle = KF.X[0][0]

    PF_x_angle_list.append(PF.x)
    KF_x_angle_list.append(x_angle)
    #print(x_angle_array)

    packet_number += 1
    print(packet_number)

plt.figure()
#plt.hist(PF_x_angle_list)
plt.plot([i for i in range(len(PF_x_angle_list))],PF_x_angle_list,label = 'PF')
plt.plot([i for i in range(len(KF_x_angle_list))],KF_x_angle_list,label='KF')
plt.plot([i for i in range(len(KF_x_angle_list))], [true_x_angle+90]*len(KF_x_angle_list),label='True x angle')
plt.xlabel('Packet id')
plt.ylabel('Angle(degree)')
plt.legend()
plt.grid()
plt.show()