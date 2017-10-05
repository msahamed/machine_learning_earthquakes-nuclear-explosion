
from paraview.simple import *
import numpy as np
import os
import csv
import pandas as pd

# set directories to get data
sub_path_high = "high_resolution_400m/"
sub_path_low = "low_resolution_800m/"
path = "/Volumes/Mini/results/energy_balance_new/" + sub_path_high
sub_file_name_low = "_800_new."
number_of_files = 2

# set models to extract data
# model_name = ['diffusion', 'energy', 'fixed', 'variable', 'deviatoric']
model_name = ['energy']
field_name = ["temperature", "density", "stress XX", "stress XZ", "stress ZZ"]
num_of_models = len(model_name)

# define geophysical parameters
K = 50e9 # bulk modulus
mu = 30e9 # Shear Modulus
alpha = K - (2.0*mu)/3.0 # Lame's constant 
c_p = 1000 # specific heat at constant pressure


for i in range(num_of_models):
    for j in range(1, number_of_files):
        file_name = str(model_name[i])+ '.'
        File_name = path + file_name + str(format(j, '06d')) + ".vtu"
        File = XMLUnstructuredGridReader(FileName=[File_name])
        cellDatatoPointData = CellDatatoPointData(Input=File)
        DataSliceFile = paraview.servermanager.Fetch(cellDatatoPointData)
        new  = DataSliceFile.GetPointData()
        print(new)
        for i in range(new):

        # get depth coordinates
        # numper_of_points = DataSliceFile.GetNumberOfPoints()
        # depth = np.zeros(numper_of_points)
        # xcord = np.zeros(numper_of_points)
        # for kk in range(numper_of_points):
        #     coord = DataSliceFile.GetPoint(kk)
        #     depth[kk] = coord[1]
        #     xcord[kk] = coord[0]
        
        # get the desired fields

        f_array = np.stack(data_array, axis=-1)

        # temp = np.array(new.GetArray(field_name[0]))
        # den = np.array(new.GetArray(field_name[1]))
        # sxx = np.array(new.GetArray(field_name[2]))
        # sxz = np.array(new.GetArray(field_name[3]))
        # szz = np.array(new.GetArray(field_name[4]))
        # vp = np.sqrt((K + (1.333*mu))/den)
        # vs = np.sqrt((alpha + (0.5 * mu))/den)
        # vp_vs = vp/vs
        # t_energy = den * c_p * temp

        f_array = np.transpose(np.array([temp, den, vp, vs, vp_vs, t_energy, depth, xcord, sxx, sxz, szz]))
        df = pd.DataFrame(f_array)
        df.to_csv('stress_data.csv', mode='a', index=False, header=True)
        # with open('stress_data.csv', 'wb') as csvfile:
        #     csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        print("Finished appending data of : " + str(model_name[i]) + "_timestep_" + str(j))

print("Finished shuffling of the data")

