from kalmanfilter2 import kalmanfilter
import numpy as np
from readdata2 import getdata
import matplotlib.pyplot as plt


# from MagnetometerTEST import calibrate, calculate_bearing


def calculate_bearing(x, y):
    """
    Calculate the bearing (angle) from the x and y axis
    :param x: x-axis
    :param y: y-axis
    """
    radian = np.arctan2(y, x)
    # Convert radian to degree
    degree = np.degrees(radian)
    # Since we are measuring relative to North, we need to subtract 90 degrees
    degree -= 90
    # Ensure the degree is within [0, 360)
    degree[degree < 0] += 360
    return degree


def calibrate(x,y):
    """
    Calibrate the magnetometer
    :param x: x-axis
    :param y: y-axis
    """
    offset_x = (max(x) + min(x)) / 2
    offset_y = (max(y) + min(y)) / 2
    #offset_z = (max(z) + min(z)) / 2

    x_cal = [i - offset_x for i in x]
    y_cal = [i - offset_y for i in y]
    #z_cal = [i - offset_z for i in z]


    return x_cal,y_cal#,z_cal


def deg2rad(deg):
    """
    Convert degrees to radians
    :param deg: degrees
    """
    return deg * np.pi / 180


# def FindClosestPoints(estimates:np.ndarray, groundtruth:np.ndarray)->tuple[np.ndarray, float]:
# 	"""
# 	Find the closest points between the estimates and the ground truth
# 	:param estimates: The estimates
# 	:param groundtruth: The ground truth
# 	"""
# 	closestPoints = np.zeros((groundtruth.shape[1], 2))
# 	errorOfApproximation = np.zeros((groundtruth.shape[1], 1))

# 	for i in range(groundtruth.shape[1]):
# 		# Find the closest point
# 		closestPoint = np.argmin(np.linalg.norm(estimates[:,:1] - groundtruth[:, i], axis=1))
# 		closestPoints[i, :] = estimates[closestPoint, :][:2]
# 		errorOfApproximation[i] = np.linalg.norm(estimates[closestPoint, :][:2] - groundtruth[:, i][:2])

# 	# calculate mean square error, mse
# 	mse = np.mean(np.linalg.norm(errorOfApproximation, axis=1))
# 	return closestPoints, mse


def FindClosestPoints(estimates:np.ndarray, groundtruth:np.ndarray)->tuple[np.ndarray, float]:
    """
    Find the closest points between the estimates and the ground truth
    :param estimates: The estimates
    :param groundtruth: The ground truth
    """
    closestPoints = np.zeros((groundtruth.shape[0], 2))
    errorOfApproximation = np.zeros((groundtruth.shape[0], 1))
    errorVector = np.zeros((groundtruth.shape[0], 3))

    for i in range(groundtruth.shape[0]):
        # Find the closest point
        closestPoint = np.argmin(np.linalg.norm(estimates[:,:2] - groundtruth[i, :2], axis=1))
        closestPoints[i, :] = estimates[closestPoint, :][:2]
        errorOfApproximation[i] = np.linalg.norm(estimates[closestPoint, :][:2] - groundtruth[i, :][:2])
        # x,y error
        errorVector[i, :2] = estimates[closestPoint, :2] - groundtruth[i,:2]

        # theta error, [rad]
        errorVector[i, 2] = estimates[closestPoint, 2] - groundtruth[i,2] * np.pi / 180

    # calculate mean square error, mse
    mse = np.mean(np.linalg.norm(errorOfApproximation, axis=1))
    earthCircle = 40075000 #meters
    m2deg = 360 / earthCircle # Convert meters to degrees (latitude)


    mse_x = np.mean(np.linalg.norm(errorVector[:,0]/m2deg))
    mse_y = np.mean(np.linalg.norm(errorVector[:,1]/m2deg))
    mse_theta = np.mean(np.linalg.norm(errorVector[:,2]))





    return closestPoints, mse, mse_x, mse_y, mse_theta, errorVector







if __name__ == "__main__":
    useGPScompass = False


    # for Mac computer use pathType = "1"
    acc_data,gyr_data,gps_data, magn_data = getdata("1") #extracing measurment data acc,gyr
    gps_data_NoNoise = np.loadtxt(r"simulatedGpsPointsNoNoise", skiprows= 1, delimiter=',')

    # gps_data = np.loadtxt(r"simulatedGpsPoints", skiprows= 1, delimiter=',')

    # for Windows computer use pathType = "2"
    # acc_data,gyr_data,gps_data, magn_data = getdata("2") #extracing measurment data acc,gyr


    groundtruth = np.array(
            [[59.347720, 18.058821], [59.346149, 18.060350], [59.346494, 18.061559], [59.348054, 18.060087]])
    groundtruth = np.vstack([groundtruth, groundtruth[0]])


    performUpdate = True # if true, perform update step, else only predict step


    acc_data, gyr_data, gps_data, magn_data = acc_data[10:,:],gyr_data[10:,:],gps_data[1:,:], magn_data[10:,:]
    gps_data[:,5] = np.pi/180 * gps_data[:,5]


    #---------------------Kalman filter uncertaunty parameters - to be changed and optimized---------------------

    #Q = np.array([[20, 0, 0], [0, 10**2, 0], [0, 0, deg2rad(10) ]])     # measurment covariacne matrix   bigger Q  => less certainty in measurment
    #R = np.array([[0.1**2, 0, 0], [0, 0.5**2, 0], [0, 0, deg2rad(1)  ]])     # uncertainty in model           bigger R => less certainty in motion model
    #sigma0 = np.array([[5, 0, 0], [0, 5, 0], [0, 0, deg2rad(5)  ]])# initial uncertainty in state

    if useGPScompass:
        #OPTIMIZED VALUES FOR GPS
        [Q11,Q22,Q33,R11,R22,R33,Sigma11,Sigma22,Sigma33] = \
            [2.00000210e+01, 1.00000016e+02, 1.72674728e-01, 3.74359136e-03,
             2.29598037e-01, 2.81146856e-02, 4.99992937e+00, 4.99997595e+00,
             8.72664668e-02]
    else:
        #OPTIMIZED VALUES FOR COMPASS
        [Q11, Q22, Q33, R11, R22, R33, Sigma11, Sigma22, Sigma33] = \
            [2.11582135e+01,  9.98928868e+01,  4.11809903e-01,  1.04821464e+00,
            7.70688459e-01, - 4.08331829e-01,  5.00000000e+00,  5.00102375e+00,
            8.74170114e-02]

    Q = np.array([[Q11, 0, 0], [0, Q22, 0],
                  [0, 0, Q33]])  # measurment covariacne matrix   bigger Q  => less certainty in measurment
    R = np.array([[R11, 0, 0], [0, R22, 0],
                  [0, 0, R33]])  # uncertainty in model           bigger R => less certainty in motion model
    sigma0 = np.array([[Sigma11, 0, 0], [0, Sigma22, 0], [0, 0, Sigma33]])  # initial uncertainty in state


    #------------------------------------------------------------------------------------------------------------



    # Set initial values, starting point
    x0 = gps_data[0,1]       #begin longitude
    y0 = gps_data[0,2]       #begin latitude
    # print("start: ", x0,y0)
    # theta0 = gps_data[0,5]   #begin angle

    if useGPScompass == True:
        #angle = np.zeros(shape= np.shape(magn_data[:,1]))

        angle = gps_data[:, 5]
        angle = np.repeat(angle, 10)

        theta0 = angle[0]


    else:
        # calibrate magnetometer, this is to have north as 0 degrees
        x_cal, y_cal = calibrate(magn_data[:,1], magn_data[:,2])
        angle = np.pi/180 * calculate_bearing(x_cal,y_cal)
        #angle = gps_data[:,5]

        # plot the angle from magnetometer
        plt.figure()
        plt.plot(magn_data[:, 0], angle, color="green", label = "Angle")
        plt.title("Angle from magnetometer")

        theta0 = angle[0]   #begin angle



    #----------- iteration limit (for debug purposes)-----------------
    # iter = 3850

    # iterate over the data, to the closest 10th data point
    # example: if acc_data.shape[0] = 3858, then iter = 3850
    iter = (acc_data.shape[0]//10 *10) if acc_data.shape[0] > magn_data.shape[0] else (magn_data.shape[0]//10 *10)
    #---------------------------------------------------------------

    # time step, set from the app used to collect data at frequency 10Hz
    frequency = 10
    dt = 1/frequency

    # define a threshold for velocity to be considered walking
    movementThreshold = 0.2


    #--------- memory variables to save the state --------------------
    stateVectorPredict = np.zeros((iter, 3))
    stateVectorUpdate = np.zeros((int(iter*dt)-1, 3))

    stateUncertaintyPredict = np.zeros((iter, 3, 3))
    stateUncertaintyUpdate = np.zeros((int(iter*dt)-1, 3, 3))

    stateVectorPredictandUpdate = np.zeros((iter, 3))

    #-----------------------------------------------------------------


    # create the kalman filter object
    kf = kalmanfilter(Q, R, x0, y0, theta0, sigma0)


    # iterate over the data
    for i in range(iter):
        a_z = acc_data[i,2] #acceleration in z-axis (z is up)
        w_z = gyr_data[i,3] # angular velocity around z-axis
        v = gps_data[i//frequency,4] #walkingspeed(a_z)

        stepdetect = a_z > movementThreshold or a_z < -movementThreshold   #detect walking or not
        v = stepdetect * v #if walking, use gps speed, else use 0

        # calculate the belief in predict step, predicted state and uncertainty
        # x_t = kf.Calcg(dt,w_z,v)



        #print("i:", i, w_z,v,"longitud latitude;", gps_data[i,2], gps_data[i,1])

        # calculate the belief in predict step, predicted state and uncertainty
        my,sigma = kf.predict(dt,w_z,v)
        # print(i, "predict:",my)

        # save the belief
        stateVectorPredict[i, :] = my
        stateUncertaintyPredict[i, :, :] = sigma

        # if iteration corresponds to measurement frequency, perform update step, obs. not if i == 0 (no previous measurement)
        if i % frequency == 0 and i != 0 and performUpdate:
            # print("perform update", "measurment recieved: ", z_t)
            # get measurment


            if useGPScompass == True:
                # GPS ANGLE
                angle[i] = gps_data[i//frequency+1,5]
                z_t = np.array([gps_data[i//frequency+1,1],gps_data[i//frequency+1,2],angle[i]]) #kf.CalcH() @
            else:
                # MAGNETOMETER ANGLE
                z_t = np.array([gps_data[i // frequency + 1, 1], gps_data[i // frequency + 1, 2],angle[i]])  # kf.CalcH() @


            # calculate Kalman gain
            K_t = kf.CalcKalmanGain()
            # calculate the error
            error = kf.CalcError(z_t)


            # print("error:", error)

            # update the belief
            my,sigma = kf.Update(K_t, error)
            # print(my)

            # save the updated belief
            stateVectorUpdate[(i // 10)-1, :] = my
            stateUncertaintyUpdate[(i // 10)-1, :, :] = sigma

        stateVectorPredictandUpdate[i, :] = my


        #print(my)

    # closestPoints, mse = FindClosestPoints(stateVectorPredictandUpdate, gps_data)
    # print("mse: ", mse)
    # mse = np.mean(np.linalg.norm(stateVectorUpdate - gps_data[1:-2, [1,2,5]], axis=1))
    # print("mse: ", mse)

    _, _, mse_x, mse_y, mse_theta, errorVector = FindClosestPoints(stateVectorUpdate, gps_data_NoNoise)
    print("mse x: ", np.round(mse_x, 4), "\nmse y: ", np.round(mse_y, 4), "\nmse theta: ", np.round(mse_theta, 4))





    #---------------------Plotting-------------------




    # plot the error

    earthCircle = 40075000 #meters
    m2deg = 360 / earthCircle # Convert meters to degrees (latitude)


    # 3 figure plot
    plt.figure()
    plt.subplot(311)
    # error in x, expressed in meters
    plt.plot(errorVector[:,0]/m2deg)
    plt.title("Error in Lambda")

    plt.xlabel("Iteration")
    plt.ylabel("Error [m]")
    # plt.legend()

    plt.subplot(312)
    # error in y, expressed in meters
    plt.plot(errorVector[:,1]/m2deg)
    plt.xlabel("Iteration")
    plt.ylabel("Error [m]")
    plt.title("Error in Phi")
    # plt.legend()

    plt.subplot(313)
    # error in theta, expressed in degrees
    plt.plot(errorVector[:,2])
    plt.xlabel("Iteration")
    plt.ylabel("Error [°]")
    plt.title("Error in theta")
    # plt.legend()


    # Adjust the space between subplots
    plt.subplots_adjust(hspace = 1)











    if True:
        # plot the results

        plt.figure(figsize=(10, 6))


        # Plot first set of x and y vectors in red color with dots
        plt.scatter(stateVectorPredict[:iter, 1], stateVectorPredict[:iter, 0], color='red', label='Predicted state')

        # Plot second set of x and y vectors in blue color with dots
        if performUpdate:
            plt.scatter(stateVectorUpdate[:iter, 1], stateVectorUpdate[:iter, 0], color='blue', label='Updated state')

        plt.scatter(gps_data[:iter//frequency+1, 2], gps_data[:iter//frequency+1, 1], color='green', label='GPS data from phone')


        plt.plot(groundtruth[:, 1], groundtruth[:, 0], color="orange", label = "Ground Truth")

        plt.legend()
        plt.ticklabel_format(useOffset=False)
        # title
        # plt.title("Kalman filter prediction")


        plt.figure()
        plt.scatter(gps_data[:, 2], gps_data[:, 1], label="Measured")
        plt.title("Location data from phone")


        # plt.show()


        plt.figure()
        plt.scatter(gps_data[:, 2], gps_data[:, 1], label="Measured")
        plt.plot(groundtruth[:, 1], groundtruth[:, 0], color="orange", label = "Ground Truth")
        plt.title("Measured data and ground truth")
        plt.legend()


        plt.figure()

        azimuthgroundtruth = np.zeros(shape=(np.shape(magn_data[:iter,0]))) #groundtruth azimuth
        azimuthgroundtruth[0:1350] = 153.6
        azimuthgroundtruth[1350:1930] = 60.8
        azimuthgroundtruth[1930:3210] = 333.6
        azimuthgroundtruth[3210:] = 240.8

        plt.plot(magn_data[:iter, 0], np.degrees(angle[:iter]), color='green', label='Compass angle measurment')
        plt.plot(magn_data[:iter, 0], np.degrees(stateVectorPredict[:iter,2]), color='blue', label='GPS Update angle')
        plt.plot(magn_data[:iter, 0], azimuthgroundtruth, color='orange', label='Ground truth angle')
        plt.legend()

        plt.show()

