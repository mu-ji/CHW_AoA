import numpy as np
from scipy.linalg import eig
from scipy.linalg import svd
import scipy.linalg as LA
import scipy.signal as ss

def array_response_vector(array,theta):
    N = array.shape
    v = np.exp(1j*2*np.pi*array*np.sin(theta))
    return v/np.sqrt(N)

def music(CovMat,L,N,array,Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    _,V = LA.eig(CovMat)
    Qn  = V[:,L:N]
    numAngles = Angles.size
    pspectrum = np.zeros(numAngles)
    for i in range(numAngles):
        av = array_response_vector(array,Angles[i])
        pspectrum[i] = 1/LA.norm((Qn.conj().transpose()@av))
    psindB    = np.log10(10*pspectrum/pspectrum.min())
    DoAsMUSIC,_= ss.find_peaks(psindB,height=1.35, distance=1.5)
    return DoAsMUSIC,pspectrum

def esprit(iq_data1, iq_data2, iq_data3, iq_data4, num_signals):

    # 将输入信号合并为一个矩阵
    iq_data = np.column_stack((iq_data1, iq_data2, iq_data3, iq_data4))
    
    # 计算协方差矩阵
    R = np.cov(iq_data)
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eig(R)
    
    # 选择信号子空间的特征向量
    signal_eigenvectors = eigenvectors[:, np.argsort(-np.abs(eigenvalues))[:num_signals]]
    
    # 构建 Φ1 和 Φ2 矩阵
    Phi1 = signal_eigenvectors[:-1, :]
    Phi2 = signal_eigenvectors[1:, :]
    
    # 计算 ESPRIT 矩阵
    Psi = np.dot(np.linalg.pinv(Phi1), Phi2)
    
    # 计算特征值
    esprit_eigenvalues = np.linalg.eigvals(Psi)
    
    # 计算频率(以 Hz 为单位)
    frequencies = np.angle(esprit_eigenvalues) / (2 * np.pi)
    
    return frequencies

