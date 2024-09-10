import numpy as np
import scipy.signal as ss
import scipy.linalg as LA
import matplotlib.pyplot as plt
 
derad = np.pi / 180
radeg = 180 / np.pi
 
def awgn(x, snr):
    spower = np.sum((np.abs(x) ** 2)) / x.size
    x = x + np.sqrt(spower / snr) * (np.random.randn(x.shape[0], x.shape[1]) + 1j * np.random.randn(x.shape[0], x.shape[1]))
    return x
 
 
def MUSIC(K,d,theta,snr,n):
 
    iwave = theta.size
    A = np.exp(-1j*2*np.pi*d.reshape(-1,1)@np.sin(theta*derad))
    S = np.random.randn(iwave,n)
    X = A@S
    X = awgn(X,snr)
    print(X.shape)
    Rxx = X@(X.conj().T)/n
    D,EV = LA.eig(Rxx)
    index = np.argsort(D)
    EN = EV.T[index].T[:,0:K-iwave]
 
 
 
 
    for i in range(numAngles):
        a = np.exp(-1j*2*np.pi*d.reshape(-1,1)*np.sin(Angles[i]))
        SP[i] = ((a.conj().T@a)/(a.conj().T@EN@EN.conj().T@a))[0,0]
 
    return SP
 
 
 
 
Angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
numAngles = Angles.size
d = np.arange(0,4,0.5)
theta = np.array([10,30,60]).reshape(1,-1)
SP = np.empty(numAngles,dtype=complex)
SP = MUSIC(K=8,d=d,theta=theta,snr=1,n=500)
 
 
SP = np.abs(SP)
SPmax = np.max(SP)
SP = 10 * np.log10(SP / SPmax)
x = Angles * radeg
plt.plot(x, SP)
plt.show()
 
 
 
 
 
 