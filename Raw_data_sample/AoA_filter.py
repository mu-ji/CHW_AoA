import numpy as np

def mean_filter(angle_array):
    # 去除离群值
    median = np.median(angle_array)
    mad = np.median(np.abs(angle_array - median))
    threshold = median + 1 * mad
    idx = np.where(np.abs(angle_array - median) < threshold)[0]
    cleaned_angles = [angle_array[i] for i in idx]
        # 计算加权平均
    weights = np.exp(-np.abs(cleaned_angles - np.mean(cleaned_angles)))
    weights /= np.sum(weights)
    weighted_sum = np.sum(cleaned_angles * weights)
    
    return weighted_sum


def antenna_shape_filter(angle_array):
    if np.mean(angle_array) > 45 and np.mean(angle_array) < 135:
        x = 0.8
        return x/2*angle_array[0] + (0.5-x/2)*angle_array[1] + x/2*angle_array[2] + (0.5-x/2)*angle_array[3]
    else:
        x = 0.2
        return x/2*angle_array[0] + (0.5-x/2)*angle_array[1] + x/2*angle_array[2] + (0.5-x/2)*angle_array[3]
    
class Kalman_Filter:
    def __init__(self,height=1,anchor_x=0,anchor_y=0,dt=0.1):
        self.X = np.zeros((1, 1))                             #Esitimation 1*1 (x)

        self.A = np.array([[1]])                     #Decided by the height, anchor_x, and anchor_y if mapping from angle space to cartesian space. For now is from angle space to angle sapce
        #self.B = np.array([[0, 0], [0, 0]])                     #if stable situation, B is zero matrix
        #self.U = np.array([0, 0]).T                             #if stable situation, U is zero vector
        self.H = np.array([[1],
                           [1],
                           [1],
                           [1],
                           [1],
                           [1],
                           [1],
                           [1]])                                     
        self.dt = 0                                           
        self.Q = np.diag([0.1])                                  
        self.R = np.diag([5,5,5,5,5,5,5,5])                            
        self.P = np.eye(1)                                         
    
    def predict(self):
        self.X = np.dot(self.A, self.X)# + np.dot(self.B, self.U)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.X)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        S = S + 1e-6*np.eye(S.shape[0])
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.X = self.X + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def update_R(self, earlier_measurement_matrix):
        if earlier_measurement_matrix.shape == (1, 8):
            return
        else:
            for i in range(len(self.R)):
                self.R[i][i] = np.std(earlier_measurement_matrix[:,i])
    
