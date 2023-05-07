import os
import sys

import math
import numpy as np
import traci
from sumolib import checkBinary

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('please declare environment variable "SUMO_HOME"')


class Env:
    def __init__(self, max_bv_num: int, cfg_sumo='./config/lane.sumocfg', output_path='', gui=False, seed=42) -> None:
        self.cfg_sumo = cfg_sumo
        self.gui = gui
        self.seed = seed
        self.name = 'LaneTest'
        self.agent = 'xyz'
        self.output_path = output_path
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        
        self.bv_num = 0
        self.av_pos = np.zeros(2)
        self.bv_pos = np.zeros((0, 2))
        self.av_vel = np.zeros(2)
        self.bv_vel = np.zeros((0, 2))
        self.scenario = np.zeros((0, 6))
        self.total_timestep = 0
        self.delta_t = 0
        self.road_len = 0
        self.max_bv_num = max_bv_num
        self.state_dim = (1 + max_bv_num) * 4   # x_pos, y_pos, speed, yaw
        self.action_dim = 2                     # delta_speed, delta_yaw
        self.action_range = np.zeros((2, 2))
        self.av_length = 0
        self.av_width = 0
        self.bv_length = 0
        self.bv_width = 0
        self.current_episode = 0
        
        command = [checkBinary(app), '--start', '-c', self.cfg_sumo]
        command += ['--routing-algorithm', 'dijkstra']
        command += ['--collision.action', 'remove']
        command += ['--seed', str(self.seed)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        command += ['--waiting-time-memory', '1000']
        command += ['--eager-insert','True']
        # command += ['--lanechange.duration', '0.1']
        command += ['--lateral-resolution', '0.0']
        # command += ['--tripinfo-output',self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        
        traci.start(command)
    
    def reset(self, scenario: np.ndarray) -> np.ndarray:
        """
        Add AV and BV and move them to their initial position. 
        
        Return the current state: an array of shape (4 * (1 + max_bv_num), ). 
        
        If the state does not have such a high dimension, which means bv_num < max_bv_num, 
        then it will be filled with 0 at the end. 
        """
        # test
        # traci.vehicle.add(vehID='test', routeID='straight', typeID='AV')
        # traci.vehicle.moveToXY(vehID='test', edgeID='', lane=0, x=10, y=4, angle=100)
        # traci.simulationStep()
        
        # clear all vehicles
        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.remove(vehicle)

        self.scenario = scenario.reshape((-1, 6))
        # self.scenario[:, 5] = -self.scenario[:, 5] * 180 / np.pi + 90     # process yaw to fit with sumo
        self.total_timestep = int(np.max(self.scenario[:, 0]))
        self.bv_num = int(np.max(self.scenario[:, 1]))
        self.road_len = math.ceil(np.max(self.scenario[:, 2]))
        self.bv_pos = np.zeros((self.bv_num, 2))    # reinitialize based on the number of BV
        self.bv_vel = np.zeros((self.bv_num, 2))    # reinitialize based on the number of BV
        self.delta_t = traci.simulation.getDeltaT()
        self.action_range = np.array(((-6, 2), (-30, 30))) * self.delta_t
        self.av_length = traci.vehicle.getLength('AV')
        self.av_width = traci.vehicle.getWidth('AV')
        self.bv_length = traci.vehicle.getLength('BV')
        self.bv_width = traci.vehicle.getWidth('BV')
        
        current_time = float(traci.simulation.getTime())
        
        # add AV and move it to its initial position
        av_id = 'AV.%d' % self.current_episode
        angle = -self.scenario[0, 5] * 180 / np.pi + 90
        traci.vehicle.add(vehID=av_id, routeID='straight', typeID='AV', depart=current_time)
        traci.vehicle.moveToXY(vehID=av_id, edgeID='', lane=0, x=self.scenario[0,2], y=self.scenario[0,3], 
                               angle=angle, matchThreshold=self.road_len)
        traci.vehicle.setLaneChangeMode(av_id, 0b000000000000)
        traci.vehicle.setSpeedMode(av_id, 0b100000)
        
        # add BV and move them to their initial position
        for i in range(self.bv_num):
            bv_id = 'BV.%d' % (i+1)
            angle = -self.scenario[i+1, 5] * 180 / np.pi + 90
            traci.vehicle.add(vehID=bv_id, routeID='straight', typeID='BV', depart=current_time)
            traci.vehicle.moveToXY(vehID=bv_id, edgeID='', lane=0, x=self.scenario[i+1,2], y=self.scenario[i+1,3], 
                                   angle=angle, matchThreshold=self.road_len)
            traci.vehicle.setLaneChangeMode(bv_id, 0b000000000000)
            traci.vehicle.setSpeedMode(bv_id, 0b100000)
        
        traci.simulationStep()
        while av_id not in traci.simulation.getDepartedIDList():
            traci.simulationStep()
        
        # update the state of AV and BV
        self.av_pos[0] = self.scenario[0,2]
        self.av_pos[1] = self.scenario[0,3]
        self.av_vel[0] = self.scenario[0,4]
        self.av_vel[1] = self.scenario[0,5]
        
        for i in range(self.bv_num):
            self.bv_pos[i,0] = self.scenario[i+1, 2]
            self.bv_pos[i,1] = self.scenario[i+1, 3]
            self.bv_vel[i,0] = self.scenario[i+1, 4]
            self.bv_vel[i,1] = self.scenario[i+1, 5]
        
        self.current_episode += 1

        state = self.get_state()
        return state
    
    def get_state(self) -> np.ndarray:
        """
        Return an array of shape (4 * (1 + max_bv_num), ), 
        which is flattened from an array with shape (bv_num + 1, 4), 
        where 4 columns represents x_pos, y_pos, speed, and yaw respectively. 
        
        The position of BV is the relative position to AV.
        
        The definition of yaw is different with SUMO, 
        which is 0~360, going clockwise with 0 at the 12'o clock position. 
        Here the yaw is -3.14~3.14, going counterclockwise with 0 at the 3'o clock position. 
        """
        bv_rel_dis = self.bv_pos.copy()
        bv_rel_dis -= self.av_pos   # relative distance between BV and AV

        av_state = np.concatenate((self.av_pos, self.av_vel)).reshape((1, -1))
        bv_state = np.concatenate((bv_rel_dis, self.bv_vel), axis=1)
        state = np.concatenate((av_state, bv_state), axis=0)    # shapes (bv_num+1, 4)
        state = state.reshape(-1)   # flatten
        # add 0 to the maximum dimension at the end of the state
        zero_num = (self.max_bv_num - self.bv_num) * 4
        state = np.concatenate((state, np.zeros(zero_num)))
        return state
    
    def step(self, av_action: np.ndarray, timestep: int) -> tuple[np.ndarray, float, bool, str]:
        av_id = "AV.%d" % (self.current_episode - 1)
        
        # move AV based on the input av_action and its current state
        v_x = self.av_vel[0] * np.cos(self.av_vel[1])
        v_y = self.av_vel[0] * np.sin(self.av_vel[1])
        v_x_ = (self.av_vel[0] + av_action[0]) * np.cos(self.av_vel[1] + av_action[1])
        v_y_ = (self.av_vel[0] + av_action[0]) * np.sin(self.av_vel[1] + av_action[1])
        
        x = self.av_pos[0] + (v_x + v_x_) * self.delta_t / 2
        y = self.av_pos[1] + (v_y + v_y_) * self.delta_t / 2
        traci.vehicle.moveToXY(av_id, edgeID='', lane=0, x=x, y=y, matchThreshold=self.road_len)
        
        # move BV based on the scenario data
        data = self.scenario[self.scenario[:, 0] == timestep]
        for i in range(self.bv_num):
            bv_id = 'BV.%d' % (i+1)
            angle = -data[i, 5] * 180 / np.pi + 90
            traci.vehicle.moveToXY(vehID=bv_id, edgeID='', lane=0, x=data[i,2], y=data[i,3], 
                                   angle=angle, matchThreshold=self.road_len)
        
        traci.simulationStep()
        
        # update the state of AV and BV
        self.av_pos[0] = x
        self.av_pos[1] = y
        self.av_vel[0] += av_action[0]
        self.av_vel[1] += av_action[1]
        
        for i in range(self.bv_num):
            bv_id = 'BV.%d' % (i+1)
            self.bv_pos[0] = data[i,2]
            self.bv_pos[1] = data[i,3]
            self.bv_vel[0] = data[i,4]
            self.bv_vel[1] = data[i,5]
        
        # return appropriate values based on the current state
        if not self.accident_detect():
            next_state = np.zeros(4 * (1 + self.max_bv_num), dtype=float)   # invalid state
            reward = -1
            done = True
            info = 'fail'
        else:
            next_state = self.get_state()
            if timestep == self.total_timestep:
                reward = 1
                done = True
                info = 'succeed'
            else:
                reward = 0
                done = False
                info = 'testing'
        
        return next_state, reward, done, info
    
    def accident_detect(self) -> bool:
        """
        Lane detect: Detect if the AV has partially driven out of the road boundary 
        (not considering the situation of fully driving out of the boundary). 
        
        Detection principle: 
        Calculate the minimum and maximum values of the y coordinates of the four vertices of AV 
        and compare them with the range of the road. 
        
        Collision detect: Detect if the AV has collided with BV. 
        
        Detection Principle: 
        Determine whether line segments intersect through vector cross multiplication, 
        and then determine whether two vehicles have collided. 
        Reference: https://blog.csdn.net/m0_37660632/article/details/123925503
        
        If an accident occurs, return False; else return True. 
        """
        av_vertex = np.zeros((4, 2))                # left front, right front, left rear, right rear
        bv_vertex = np.zeros((self.bv_num, 4, 2))   # left front, right front, left rear, right rear
        road_edge = (0, 12)
        av_vertex[0,0] = self.av_pos[0] - self.av_width / 2 * np.sin(self.av_vel[1])
        av_vertex[0,1] = self.av_pos[1] + self.av_width / 2 * np.cos(self.av_vel[1])
        av_vertex[1,0] = self.av_pos[0] + self.av_width / 2 * np.sin(self.av_vel[1])
        av_vertex[1,1] = self.av_pos[1] - self.av_width / 2 * np.cos(self.av_vel[1])
        av_vertex[2,0] = self.av_pos[0] - self.av_width / 2 * np.sin(self.av_vel[1]) \
                                        - self.av_length    * np.cos(self.av_vel[1])
        av_vertex[2,1] = self.av_pos[1] + self.av_width / 2 * np.cos(self.av_vel[1]) \
                                        - self.av_length    * np.sin(self.av_vel[1])
        av_vertex[3,0] = self.av_pos[0] + self.av_width / 2 * np.sin(self.av_vel[1]) \
                                        - self.av_length    * np.cos(self.av_vel[1])
        av_vertex[3,1] = self.av_pos[1] - self.av_width / 2 * np.cos(self.av_vel[1]) \
                                        - self.av_length    * np.sin(self.av_vel[1])
        
        for i in range(self.bv_num):
            bv_vertex[i,0,0] = self.bv_pos[i,0] - self.bv_width / 2 * np.sin(self.bv_vel[i,1])
            bv_vertex[i,0,1] = self.bv_pos[i,1] + self.bv_width / 2 * np.cos(self.bv_vel[i,1])
            bv_vertex[i,1,0] = self.bv_pos[i,0] + self.bv_width / 2 * np.sin(self.bv_vel[i,1])
            bv_vertex[i,1,1] = self.bv_pos[i,1] - self.bv_width / 2 * np.cos(self.bv_vel[i,1])
            bv_vertex[i,2,0] = self.bv_pos[i,0] - self.bv_width / 2 * np.sin(self.bv_vel[i,1]) \
                                                - self.bv_length    * np.cos(self.bv_vel[i,1])
            bv_vertex[i,2,1] = self.bv_pos[i,1] + self.bv_width / 2 * np.cos(self.bv_vel[i,1]) \
                                                - self.bv_length    * np.sin(self.bv_vel[i,1])
            bv_vertex[i,3,0] = self.bv_pos[i,0] + self.bv_width / 2 * np.sin(self.bv_vel[i,1]) \
                                                - self.bv_length    * np.cos(self.bv_vel[i,1])
            bv_vertex[i,3,1] = self.bv_pos[i,1] - self.bv_width / 2 * np.cos(self.bv_vel[i,1]) \
                                                - self.bv_length    * np.sin(self.bv_vel[i,1])

        # lane detect
        if (np.max(av_vertex[:,1]) > road_edge[1]) or (np.min(av_vertex[:,1]) < road_edge[0]):
            return False
        
        # TODO: collision detect
        
