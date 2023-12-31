import os
import sys
import math
from shapely.geometry import Polygon
import numpy as np
import traci
from sumolib import checkBinary

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit('please declare environment variable SUMO_HOME')


class Env:
    def __init__(self, max_bv_num: int, cfg_sumo='./config/lane.sumocfg', gui=False, delay=0, 
                 reward_type='r3', bv_control='data', seed=42) -> None:
        self.cfg_sumo = cfg_sumo
        self.gui = gui
        self.reward_type = reward_type
        self.bv_control = bv_control
        self.seed = seed
        self.name = 'LaneTest'
        self.agent = 'xyz'
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
        command += ['--lanechange.duration', '1.5']
        command += ['--lateral-resolution', '0.0']
        command += ['--delay', str(delay)]
        traci.start(command)
        
        self.av_pos = np.zeros(2)
        self.av_vel = np.zeros(2)
        self.max_bv_num = max_bv_num
        self.state_dim = (1 + max_bv_num) * 4   # x_pos, y_pos, speed, yaw of each vehicle
        self.action_dim = 2                     # delta_speed, delta_yaw
        self.delta_t = traci.simulation.getDeltaT()
        self.av_accel = 5.88
        self.av_decel = -7.84
        self.action_range = np.array(((self.av_decel, self.av_accel), (-np.pi/6, np.pi/6))) * self.delta_t
        self.current_episode = 0
    
    def reset(self, scenario: np.ndarray) -> np.ndarray:
        """
        Add AV and BV and move them to their initial position. \n
        Return the current state: an array of shape (4 * (1 + max_bv_num), ). \n
        If the state does not have such a high dimension, which means bv_num < max_bv_num, 
        then it will be filled with 0 at the end. 
        """
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
        traci.vehicle.add(vehID=av_id, routeID='straight', typeID='AV', 
                          depart=current_time, departSpeed=self.scenario[0,4])
        traci.vehicle.moveToXY(vehID=av_id, edgeID='', lane=0, x=self.scenario[0,2], y=self.scenario[0,3], 
                               angle=angle, matchThreshold=self.road_len)
        traci.vehicle.setLaneChangeMode(av_id, 0b000000000000)
        traci.vehicle.setSpeedMode(av_id, 0b100000)
        
        # add BV and move them to their initial position
        for i in range(self.bv_num):
            bv_id = 'BV.%d' % (i+1)
            angle = -self.scenario[i+1, 5] * 180 / np.pi + 90
            traci.vehicle.add(vehID=bv_id, routeID='straight', typeID='BV', 
                              depart=current_time, departSpeed=self.scenario[i+1,4])
            traci.vehicle.moveToXY(vehID=bv_id, edgeID='', lane=0, x=self.scenario[i+1,2], y=self.scenario[i+1,3], 
                                   angle=angle, matchThreshold=self.road_len)
            traci.vehicle.setLaneChangeMode(bv_id, 0b000000000000)
            traci.vehicle.setSpeedMode(bv_id, 0b100000)
        
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
        which is flattened from an array with shape (1 + max_bv_num, 4), 
        where 4 columns represents x_pos, y_pos, speed, and yaw respectively. 
        
        The position of BV is the relative position to AV.
        
        The definition of yaw is different with SUMO, 
        which is measured in degrees, going clockwise with 0 at the 12'o clock position. 
        Here the yaw is measured in radians, going counterclockwise with 0 at the 3'o clock position. 
        """
        bv_rel_dis = self.bv_pos.copy() # deep copy
        bv_rel_dis -= self.av_pos   # relative distance between BV and AV
        
        av_state = np.concatenate((self.av_pos, self.av_vel)).reshape((1, -1))
        bv_state = np.concatenate((bv_rel_dis, self.bv_vel), axis=1)
        state = np.concatenate((av_state, bv_state), axis=0)    # shapes (1+bv_num, 4)
        
        # add invalid states to the maximum dimension at the end of the state
        empty_num = self.max_bv_num - self.bv_num
        # state = np.block([[state], [1000 * np.ones((empty_num, 2)), np.zeros((empty_num, 2))]])
        state = np.block([[state], [np.zeros((empty_num, 4))]])
        state = state.reshape(-1)   # flatten
        
        return state
    
    def step(self, av_action: np.ndarray, timestep: int, need_reward=True) -> tuple[np.ndarray, float, bool, str]:
        av_id = 'AV.%d' % (self.current_episode - 1)
        
        # # extend the action to the actual range
        # av_action[0] = (self.action_range[0,1] + self.action_range[0,0] + \
        #                (self.action_range[0,1] - self.action_range[0,0]) * av_action[0]) / 2.0
        # av_action[1] = (self.action_range[1,1] + self.action_range[1,0] + \
        #                (self.action_range[1,1] - self.action_range[1,0]) * av_action[1]) / 2.0
        
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
        
        if self.bv_control == 'data':
            # move BV based on the scenario data
            data = self.scenario[self.scenario[:, 0] == timestep]
            for i in range(self.bv_num):
                bv_id = 'BV.%d' % (i+1)
                angle = -data[i,5] * 180 / np.pi + 90
                traci.vehicle.moveToXY(vehID=bv_id, edgeID='', lane=0, x=data[i,2], y=data[i,3], 
                                    angle=angle, matchThreshold=self.road_len)
        elif self.bv_control == 'sumo':
            # move BV based on SUMO
            pass
        
        traci.simulationStep()
        
        # update the state of AV and BV
        self.av_pos[0] = x
        self.av_pos[1] = y
        self.av_vel[0] += av_action[0]
        self.av_vel[1] += av_action[1]
        
        for i in range(self.bv_num):
            bv_id = 'BV.%d' % (i+1)
            if self.bv_control == 'data':
                self.bv_pos[i,0] = data[i,2]
                self.bv_pos[i,1] = data[i,3]
                self.bv_vel[i,0] = data[i,4]
                self.bv_vel[i,1] = data[i,5]
            elif self.bv_control == 'sumo':
                self.bv_pos[i,0] = traci.vehicle.getPosition(bv_id)[0]
                self.bv_pos[i,1] = traci.vehicle.getPosition(bv_id)[1]
                self.bv_vel[i,0] = traci.vehicle.getSpeed(bv_id)
                self.bv_vel[i,1] = (90 - traci.vehicle.getAngle(bv_id)) * np.pi / 180
        
        no_accident = self.accident_detect()
        reward = self.get_reward(self.reward_type, no_accident) if need_reward else 0.0
        next_state = self.get_state()
        
        # return appropriate values based on the current state
        if no_accident:
            if timestep == self.total_timestep:
                done = True
                info = 'succeed'
            else:
                done = False
                info = 'testing'
        else:
            done = True
            info = 'fail'
        
        return next_state, reward, done, info
    
    def accident_detect(self) -> bool:
        """
        ## Lane detect
        Detect if the AV has driven out of the road boundary. 
        
        Detection principle: 
        Calculate the minimum and maximum values of the y coordinates of AV's four vertices 
        and compare them with the range of the road. 
        
        ## Collision detect
        Detect if the AV has collided with BV. 
        
        Detection Principle: 
        Determine whether line segments intersect through vector cross multiplication, 
        and then determine whether two vehicles have collided. 
        Reference: https://blog.csdn.net/m0_37660632/article/details/123925503
        #### Update: 
        Directly call the Shapely library for judgment. 
        
        ## Return
        If an accident occurs, return FALSE; else return TRUE. 
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
        
        # collision detect
        # def xmult(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        #     """Finding the cross product of two-dimensional vector ab and vector cd. """
        #     vectorAx = b[0] - a[0]
        #     vectorAy = b[1] - a[1]
        #     vectorBx = d[0] - c[0]
        #     vectorBy = d[1] - c[1]
        #     return (vectorAx * vectorBy - vectorAy * vectorBx)
        
        # for i in range(self.bv_num):
        #     for p in range(4):
        #         c, d = av_vertex[p-1], av_vertex[p]
        #         for q in range(4):
        #             a, b = bv_vertex[i, q-1], bv_vertex[i, q]
        #             if (xmult(c,d,c,a) * xmult(c,d,c,b) < 0) and (xmult(a,b,a,c) * xmult(a,b,a,d) < 0):
        #                 return False
        
        av_rectangle = Polygon(av_vertex)
        for i in range(self.bv_num):
            bv_rectangle = Polygon(bv_vertex[i])
            if av_rectangle.intersects(bv_rectangle):
                return False
        
        return True
    
    def get_reward(self, reward_type: str, no_accident: bool) -> float:
        if reward_type == 'r0':
            r_col = 0 if no_accident else -40
            r_vel = 0.8 * (2 * self.av_vel[0] - self.av_max_speed) / self.av_max_speed
            r_yaw = -abs(self.av_vel[1]) / (np.pi / 6)
            reward = r_col + r_vel + r_yaw
        elif reward_type == 'r1':
            r_col = 0 if no_accident else -40
            r_vel = 5 * self.av_vel[0] / self.av_max_speed
            if r_vel > 5:
                r_vel = 10 - r_vel
            r_yaw = -0.5 * ((self.av_vel[1]) ** 2)
            reward = r_col + r_vel + r_yaw
        elif reward_type == 'r2':
            r_col = 0 if no_accident else -40
            
            v_des = 30
            r_v = math.exp(-((self.av_vel[0] - v_des)**2) / 10) - 1
            
            width = traci.lane.getWidth('lane0')
            if self.av_pos[1] > 2 * width:
                y_des = 2.5 * width
            elif self.av_pos[1] < width:
                y_des = 0.5 * width
            else:
                y_des = 1.5 * width
            r_y = math.exp(-((self.av_pos[1] - y_des)**2) / 10) - 1
            
            d_lead = traci.lane.getLength('lane0')
            for i in range(self.bv_num):
                if (self.bv_pos[i, 1] < self.av_pos[1] + self.av_width / 2) and \
                   (self.bv_pos[i, 1] > self.av_pos[1] - self.av_width / 2):
                    if (self.bv_pos[i, 0] - self.av_pos[0] > 0) and (self.bv_pos[i, 0] - self.av_pos[0] < d_lead):
                        d_lead = self.bv_pos[i, 0] - self.av_pos[0]
            
            d_safe = 50
            r_x = math.exp(-((d_lead - d_safe)**2) / (10 * d_safe)) - 1 if d_lead < d_safe else 0
            
            reward = r_col + r_v + r_y + r_x
        elif reward_type == 'r3':
            r_col = 0 if no_accident else -40
            r_vel = 0.8 * (2 * self.av_vel[0] - self.av_max_speed) / self.av_max_speed
            r_yaw = -abs(self.av_vel[1]) / (np.pi / 6)
            
            dis = traci.lane.getLength('lane0') * np.ones(3)
            width = traci.lane.getWidth('lane0')
            for i in range(self.bv_num):
                if 0 < self.bv_pos[i,1] and self.bv_pos[i,1] < width:               # BV in lane 0
                    dis[0] = self.bv_pos[i, 0] - self.av_pos[0]
                elif width <= self.bv_pos[i,1] and self.bv_pos[i,1] < 2*width:      # BV in lane 1
                    dis[1] = self.bv_pos[i,0] - self.av_pos[0]
                elif 2*width <= self.bv_pos[i,1] and self.bv_pos[i,1] < 3*width:    # BV in lane 2
                    dis[2] = self.bv_pos[i, 0] - self.av_pos[0]
            best_lane = np.argwhere(dis == np.max(dis)).flatten()
            av_lane = int(self.av_pos[1] / width)
            r_lane = 2 if av_lane in best_lane else 0
            
            reward = r_col + r_vel + r_yaw + r_lane
        
        return reward
    
    def close(self) -> None:
        traci.close()
