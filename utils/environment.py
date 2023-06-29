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
    sys.exit('please declare environment variable SUMO_HOME')


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
        
        self.av_pos = np.zeros(2)
        self.av_vel = np.zeros(2)
        self.max_bv_num = max_bv_num
        self.state_dim = (1 + max_bv_num) * 4   # x_pos, y_pos, speed, yaw
        self.action_dim = 2                     # delta_speed, delta_yaw
        self.delta_t = traci.simulation.getDeltaT()
        self.av_accel = -7.84
        self.av_decel = 5.88
        self.action_range = np.array(((self.av_decel, self.av_accel), (-np.pi/6, np.pi/6))) * self.delta_t
        self.current_episode = 0
    
    def reset(self, scenario: np.ndarray) -> np.ndarray:
        """
        Add AV and BV and move them to their initial position. 
        
        Return the current state: an array of shape (4 * (1 + max_bv_num), ). 
        
        If the state does not have such a high dimension, which means bv_num < max_bv_num, 
        then it will be filled with 0 at the end. 
        """
        #################### test ####################
        # traci.vehicle.add(vehID='AV', routeID='straight', typeID='AV')
        # traci.vehicle.moveToXY(vehID='AV', edgeID='', lane=0, x=10, y=0, angle=100)
        # traci.vehicle.add(vehID='BV', routeID='straight', typeID='BV')
        # traci.vehicle.moveToXY(vehID='BV', edgeID='', lane=0, x=11, y=4, angle=60)
        # traci.simulationStep()
        # self.bv_num = 1
        # self.av_length = traci.vehicle.getLength('AV')
        # self.av_width = traci.vehicle.getWidth('AV')
        # self.bv_length = traci.vehicle.getLength('BV')
        # self.bv_width = traci.vehicle.getWidth('BV')
        # self.av_pos = np.array((10, 0), dtype=float)
        # self.bv_pos = np.array((11, 4), dtype=float).reshape((1, -1))
        # self.av_vel = np.array((10, -0.174533))
        # self.bv_vel = np.array((10, 0.5236)).reshape((1, -1))
        # accident = self.accident_detect()
        ##############################################
        
        # clear all vehicles
        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.remove(vehicle)

        self.scenario = scenario.reshape((-1, 6))
        self.total_timestep = int(np.max(self.scenario[:, 0]))
        self.bv_num = int(np.max(self.scenario[:, 1]))
        self.road_len = math.ceil(np.max(self.scenario[:, 2]))
        self.bv_pos = np.zeros((self.bv_num, 2))
        self.bv_vel = np.zeros((self.bv_num, 2))
        
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
        
        self.av_accel = traci.vehicle.getAccel(av_id)
        self.av_decel = -traci.vehicle.getDecel(av_id)
        self.action_range = np.array(((self.av_decel, self.av_accel), (-np.pi/6, np.pi/6))) * self.delta_t
        self.av_length = traci.vehicle.getLength(av_id)
        self.av_width = traci.vehicle.getWidth(av_id)
        self.bv_length = traci.vehicle.getLength(bv_id)
        self.bv_width = traci.vehicle.getWidth(bv_id)
        self.av_max_speed = traci.vehicle.getMaxSpeed(av_id)
        
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
        which is measured in degrees, going clockwise with 0 at the 12'o clock position. 
        Here the yaw is measured in radians, going counterclockwise with 0 at the 3'o clock position. 
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
        av_id = 'AV.%d' % (self.current_episode - 1)
        
        # move AV based on the input av_action and its current state
        v_x = self.av_vel[0] * np.cos(self.av_vel[1])
        v_y = self.av_vel[0] * np.sin(self.av_vel[1])
        v_x_ = (self.av_vel[0] + av_action[0]) * np.cos(self.av_vel[1] + av_action[1])
        v_y_ = (self.av_vel[0] + av_action[0]) * np.sin(self.av_vel[1] + av_action[1])
        angle = -(self.av_vel[1] + av_action[1]) * 180 /np.pi + 90
        
        x = self.av_pos[0] + (v_x + v_x_) * self.delta_t / 2
        y = self.av_pos[1] + (v_y + v_y_) * self.delta_t / 2
        traci.vehicle.moveToXY(vehID=av_id, edgeID='', lane=0, x=x, y=y, 
                               angle=angle, matchThreshold=self.road_len)
        
        # move BV based on the scenario data
        data = self.scenario[self.scenario[:, 0] == timestep]
        for i in range(self.bv_num):
            bv_id = 'BV.%d' % (i+1)
            angle = -data[i,5] * 180 / np.pi + 90
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
            self.bv_pos[i,0] = data[i,2]
            self.bv_pos[i,1] = data[i,3]
            self.bv_vel[i,0] = data[i,4]
            self.bv_vel[i,1] = data[i,5]
        
        # reward function
        r_speed = (2 * self.av_vel[0] - self.av_max_speed) / self.av_max_speed  # encourage faster speed
        r_yaw = -abs(self.av_vel[1]) / (np.pi / 6)  # punish sharp turns
        reward = r_speed + r_yaw
        
        # return appropriate values based on the current state
        if not self.accident_detect():
            next_state = np.zeros(4 * (1 + self.max_bv_num), dtype=float)   # invalid state
            reward -= 10    # punish collision
            done = True
            info = 'fail'
        else:
            if timestep == self.total_timestep:
                next_state = np.zeros(4 * (1 + self.max_bv_num), dtype=float)   # invalid state
                done = True
                info = 'succeed'
            else:
                next_state = self.get_state()
                done = False
                info = 'testing'
        
        return next_state, reward, done, info
    
    def accident_detect(self) -> bool:
        """
        Lane detect: Detect if the AV has driven out of the road boundary. 
        
        Detection principle: 
        Calculate the minimum and maximum values of the y coordinates of AV's four vertices 
        and compare them with the range of the road. 
        
        Collision detect: Detect if the AV has collided with BV. 
        
        Detection Principle: 
        Determine whether line segments intersect through vector cross multiplication, 
        and then determine whether two vehicles have collided. 
        Reference: https://blog.csdn.net/m0_37660632/article/details/123925503
        
        If an accident occurs, return False; else return True. 
        """
        # calculate the vertex coordinates of each vehicle
        av_vertex = np.zeros((4, 2))                # left front, right front, left rear, right rear
        bv_vertex = np.zeros((self.bv_num, 4, 2))   # left front, right front, left rear, right rear
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
        road_edge = (0, 12)
        if (np.max(av_vertex[:,1]) > road_edge[1]) or (np.min(av_vertex[:,1]) < road_edge[0]):
            return False
        
        def xmult(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
            """Finding the cross product of two-dimensional vector ab and vector cd. """
            vectorAx = b[0] - a[0]
            vectorAy = b[1] - a[1]
            vectorBx = d[0] - c[0]
            vectorBy = d[1] - c[1]
            return (vectorAx * vectorBy - vectorAy * vectorBx)
        
        # collision detect
        for i in range(self.bv_num):
            for p in range(4):
                c, d = av_vertex[p-1], av_vertex[p]
                for q in range(4):
                    a, b = bv_vertex[i, q-1], bv_vertex[i, q]
                    if (xmult(c,d,c,a) * xmult(c,d,c,b) < 0) and (xmult(a,b,a,c) * xmult(a,b,a,d) < 0):
                        return False
        
        return True
    
    def close(self) -> None:
        traci.close()
