import numpy as np

from doa_py import arrays, signals
from doa_py.algorithm import music
from doa_py.algorithm import root_music
from doa_py.plot import plot_spatial_spectrum

import AoA_cal_angle

data = np.load('Raw_data_sample/0006_2_(0,0)_data.npz')
packet_number = 0
while packet_number < 200:
    packet_number += 1
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

        antenna_data_list.append(complex_signal)
        i = i + 16


    ula = arrays.UniformLinearArray(m=3, dd=0.0375)

    received_data = np.array([antenna_data_list[0],
                        antenna_data_list[1],
                        antenna_data_list[2]
                        ])

    print(packet_number)

    angle_grids = np.arange(-90, 90, 1)
    spectrum = music(
        received_data=received_data,
        num_signal=1,
        array=ula,
        signal_fre=3e8,
        angle_grids=angle_grids,
        unit="deg",
    )

    # Plot the spatial spectrum
    plot_spatial_spectrum(
        spectrum=spectrum,
        ground_truth=np.array([0]),
        angle_grids=angle_grids,
        num_signal=1,
    )