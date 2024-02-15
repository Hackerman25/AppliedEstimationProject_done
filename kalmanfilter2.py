import numpy as np
import math
from numpy.linalg import inv



class kalmanfilter:

    def __init__(self,Q:np.ndarray,R: np.ndarray,x0: float,y0: float,theta0:float,sigma0:np.ndarray):
        """
        Initialize the kalman filter
        :param Q: The measurement covariance matrix
        :param R: The uncertainty in the model
        :param x0: Initial x position (in degrees, longitude)
        :param y0: Initial y position (in degrees, latitude)
        :param theta0: Initial theta (in degrees)
        :param sigma0: Initial uncertainty
        """

        self.Q = Q
        self.R = R
        self.dt = 0
        self.x = x0
        self.y = y0
        self.theta = theta0
        self.sigma = sigma0
        self.my = np.array([x0,y0,theta0]).T
        
        earthCircle = 40075000 #meters
        self.m2deg = 360 / earthCircle # Convert meters to degrees (latitude)


    def CalcG(self,dt:float,v: float)-> np.ndarray: # State = [x;y;theta]
        """
        Calculates the G matrix, which is the Jacobian of the state transition function g
        :param dt: The time step
        :param v: The velocity
        """
        self.G = np.array([[1,0,-(dt * v) * math.cos(math.pi/2 -self.theta) *self.m2deg ],
                          [0,1,-(dt * v) * math.sin(math.pi/2 - self.theta) *self.m2deg],
                          [0,0,1]])
        return self.G
    
    def Calcg(self,dt,w_z,v):
        """
        Calculates the state transition function g, gives next state according to motion model
        :param dt: The time step
        :param w_z: The angular velocity 
        :param v: The velocity
        """

        x,y,self.theta = self.my[0], self.my[1], self.my[2]
        theta = self.theta

        theta = theta - w_z * dt   #todo: might be after x,y
        x = x + ( (dt * v) * math.sin(math.pi/2 -theta) ) *self.m2deg    # math.pi/2 - theta is to project angle from north
        y = y + ( (dt * v) * math.cos(math.pi/2 - theta)  )  *self.m2deg 



        return np.array([x,y,theta])
    

    def CalcH(self):
        """
        Calculates the H matrix, which is the Jacobian of the measurement function h
        """
        # measure x, y directly and theta is not directly measured, therefore:
        #NOTE: not ekf but kf, might have to be changed, change 3,3 to dw/dtheta
        self.H = np.array([[1,0,0],[0,1,0],[0,0,1]])  # h * [x;y;theta] theta <- kompass?

        return self.H

    def predict(self,dt:float,w_z:float,v:float):
        """
        Predicts the next state and uncertainty
        :param dt: The time step
        :param w_z: The angular velocity
        :param v: The velocity
        """

        # calc. belief according to motion model
        self.my = self.Calcg(dt,w_z,v)


        self.CalcG(dt,v)

        # calc. uncertainty, according to Kalman Filter prediction step
        self.sigma = self.R + self.G @ self.sigma @ self.G.T

        return self.my, self.sigma


    def CalcKalmanGain(self, Q = None):
        """
        Calculates the Kalman gain
        """
        
        self.CalcH()

        if Q is None:
            Q = self.Q

        # calc. Kalman gain 
        Kt = self.sigma @ self.H.T @ inv( self.H @ self.sigma @ self.H.T + Q)
        return Kt

    def CalcError(self,measurments: np.ndarray):
        """
        Calculates the error
        :param measurments: The measurments
        """

        # error = -(self.my - measurments)
        error =  measurments - self.my

        # print("measurments: ", measurments)

        return error

    def Update(self,Kt: np.ndarray,error: np.ndarray):
        """
        Updates the belief
        :param Kt: The Kalman gain
        :param error: The error
        """


        # print("innan: ", self.my)

        # update the belief
        self.sigma = self.sigma - Kt @ self.H @ self.sigma

        # print("@", "Kt: ", Kt, "error: ", error)
        
        self.my = self.my + Kt @ error

        # print("after: ", self.my)

        return self.my,self.sigma
    







    #x = x + (dt * v) * math.sin(-theta)
    #y = y + (dt * v) * math.cos(theta)
    # theta = theta + w_z * dt


def main():
    """
    For testing purposes, to check that there are no errors in kalman filter implementation
    """
    Q = np.array([[1,0,0],[0,1,0],[0,0,1]])  #measurment covariacne matrix
    R =  np.array([[1,0,0],[0,1,0],[0,0,1]]) #uncertainty in model
    x0 = 1
    y0 = 1
    theta0 = 1
    sigma0 = np.array([[1,0,0],[0,1,0],[0,0,1]])

    kf = kalmanfilter(Q,R,x0,y0,theta0,sigma0)

    #print(kf.CalcG(dt,v,theta))


    #print(kf.Calcg(dt,w_z,v))

    #print(kf.CalcH())


    #print(kf.predict(dt,w_z,v))

    #kf.CalcH()
    #print(kf.CalcKalmanGain())

    #print(kf.CalcError(np.array([0.5,0.5,0.5]).T))

    #kf.CalcH()



    #error = kf.CalcError(np.array([0.5,0.5,0.5]).T)
    #Kt = kf.CalcKalmanGain()
    #print(kf.Update(Kt,error))









if __name__ == "__main__":
    main()