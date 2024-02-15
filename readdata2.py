import csv
import numpy as np

def getdata(pathType:str = "2"):
	"""
	Reads and returns the data from the csv files
	:param pathType: The type of path to read from
		pathType = "1" => Mac path
		pathType = "2" => Windows path
	:return: The data from the csv files
	"""

	if pathType == "1": # Mac path
		file_path_acc = r"experiment3/Linear Acceleration.csv"
		file_path_gyr = r"experiment3/Gyroscope.csv"
		file_path_gps = r"experiment3/Location.csv"
		file_path_magn = r"experiment3/Magnetometer.csv"
		
	else: # windows path
		file_path_acc = r"experiment3\Linear Acceleration.csv"
		file_path_gyr = r"experiment3\Gyroscope.csv"
		file_path_gps = r"experiment3\Location.csv"
		file_path_magn = r"experiment3\Magnetometer.csv"

	# read the data from the csv files
	csv_data_acc = np.loadtxt(file_path_acc, skiprows= 1, delimiter=',')
	csv_data_gyr = np.loadtxt(file_path_gyr, skiprows= 1, delimiter=',')
	csv_data_gps = np.loadtxt(file_path_gps, skiprows=1, delimiter=',')
	csv_data_magn = np.loadtxt(file_path_magn, skiprows=1, delimiter=',')

	
	return csv_data_acc, csv_data_gyr, csv_data_gps, csv_data_magn 

