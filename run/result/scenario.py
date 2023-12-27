import os
import sys
root_path = os.getcwd()
sys.path.append(root_path)

from tqdm import trange
from math import sqrt, pi
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

from utils.scenario_lib import scenario_lib

random_seed = 42
lib = scenario_lib(path='./data/all/', npy_path='./data/all.npy')

delta_t = 0.04
bv_bv_dis, bv_av_dis, bv_av_pos_x, bv_av_pos_y, bv_yaw, bv_vel, bv_acc = [], [], [], [], [], [], []
for i in trange(lib.scenario_num):
    scenario = lib.data[i].reshape((-1, 6))
    total_timestep = int(np.max(scenario[:, 0]))
    bv_num = int(np.max(scenario[:, 1]))
    for j in range(bv_num):
        bv_av_dis.append(sqrt((scenario[0, 2] - scenario[1 + j, 2]) ** 2 + 
                              (scenario[0, 3] - scenario[1 + j, 3]) ** 2))
        bv_av_pos_x.append(scenario[1 + j, 2] - scenario[0, 2])
        bv_av_pos_y.append(scenario[1 + j, 3] - scenario[0, 3])
        for t in range(total_timestep):
            bv_yaw.append(scenario[1 + t * bv_num + j, 5] * 180 / pi)
            bv_vel.append(scenario[1 + t * bv_num + j, 4])
            if t < total_timestep - 1:
                bv_acc.append((scenario[1 + (t + 1) * bv_num + j, 4] - scenario[1 + t * bv_num + j, 4]) / delta_t)
            for k in range(bv_num):
                if k != j:
                    bv_bv_dis.append(sqrt((scenario[1 + t * bv_num + j, 2] - scenario[1 + t * bv_num + k, 2]) ** 2 + 
                                          (scenario[1 + t * bv_num + j, 3] - scenario[1 + t * bv_num + k, 3]) ** 2))


plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_bv_dis, dtype=np.float16), bins=50, alpha=0.8, density=True)
plt.title('BV-BV distance distribution')
plt.xlabel('distance(m)')
plt.ylabel('frequency')
plt.savefig('./figure/bv_bv_dis.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_av_dis, dtype=np.float16), bins=25, alpha=0.8, density=True)
plt.title('BV-AV distance distribution')
plt.xlabel('distance(m)')
plt.ylabel('frequency')
plt.xlim(-8, 175)
plt.savefig('./figure/bv_av_dis.pdf', bbox_inches='tight')

random.seed(random_seed)
index = random.sample(range(len(bv_av_pos_x)), 2000)
bv_av_pos_x = [bv_av_pos_x[i] for i in index]
bv_av_pos_y = [bv_av_pos_y[i] for i in index]
plt.figure(figsize=(9, 1.4))
plt.scatter(np.array(bv_av_pos_x, dtype=np.float16), np.array(bv_av_pos_y, dtype=np.float16), 
            s=2, alpha=0.5, label='BV')
plt.scatter(0, 0, s=20, c='r', label='AV')
plt.grid()
plt.legend()
plt.axis('equal')
plt.title('BV-AV relative position')
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.savefig('./figure/bv_av_pos.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_yaw, dtype=np.float16), bins=50, alpha=0.8, density=True)
plt.title('BV yaw distribution')
plt.xlabel(r'yaw($^\circ$)')
plt.ylabel('frequency')
plt.savefig('./figure/bv_yaw.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_vel, dtype=np.float16), bins=50, alpha=0.8, density=True)
plt.title('BV velocity distribution')
plt.xlabel('velocity(m/s)')
plt.ylabel('frequency')
plt.savefig('./figure/bv_vel.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.hist(np.array(bv_acc, dtype=np.float16), bins=50, alpha=0.8, density=True)
plt.title('BV acceration distribution')
plt.xlabel(r'acceleration(m/s$^2$)')
plt.ylabel('frequency')
plt.xlim(-9, 9)
plt.xticks(np.arange(-8, 10, 4))
plt.savefig('./figure/bv_acc.pdf', bbox_inches='tight')
