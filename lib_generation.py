import os

import numpy as np

path = 'D:\\school\\科研\\课题\\RL_Scenario_Generation-RL_scenario_generation_rk\\byH2O\\output\\'
save_path = 'D:\\school\\科研\\课题\\CL-for-Autonomous-Vehicle-Training-and-Testing\\scenario_lib\\'
subpath = os.listdir(path)
i = 0

for sp in subpath:
    p = path+sp+'\\avarrive\\'
    file_list = os.listdir(p)
    for filename in file_list:
        data = np.loadtxt(p+filename, dtype=np.float64)
        data = data[data[:, 1] > 0]
        data = data[:, [0,1,3,4,6]]

        np.savetxt(save_path+str(i)+'.csv', data, delimiter=',', header='time_step,bv_id,x_pos,y_pos,yaw', comments='')
        i += 1

    p = path+sp+'\\avcrash\\'
    file_list = os.listdir(p)
    for filename in file_list:
        data = np.loadtxt(p+filename, dtype=np.float64)
        data = data[data[:, 1] > 0]
        data = data[:, [0,1,3,4,6]]

        np.savetxt(save_path+str(i)+'.csv', data, delimiter=',', header='time_step,bv_id,x_pos,y_pos,yaw', comments='')
        i += 1
