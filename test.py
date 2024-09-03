import numpy as np

from doa_py import arrays, signals
from doa_py.algorithm import music
from doa_py.plot import plot_spatial_spectrum

# Create a 8-element ULA with 0.5m spacing
ula = arrays.UniformLinearArray(m=4, dd=0.0375)
# Create a complex stochastic signal
source = signals.ComplexStochasticSignal(fc=3e8)

# Simulate the received data
received_data = ula.received_signal(
    signal=source, snr=0, nsamples=1000, angle_incidence=np.array([0, 30]), unit="deg"
)
print(received_data[0])
# Calculate the MUSIC spectrum
angle_grids = np.arange(-90, 90, 1)
spectrum = music(
    received_data=received_data,
    num_signal=2,
    array=ula,
    signal_fre=3e8,
    angle_grids=angle_grids,
    unit="deg",
)

# Plot the spatial spectrum
plot_spatial_spectrum(
    spectrum=spectrum,
    ground_truth=np.array([0, 30]),
    angle_grids=angle_grids,
    num_signal=2,
)