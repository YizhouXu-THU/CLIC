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
    sys.exit("please declare environment variable 'SUMO_HOME'")


class Env:
    def __init__(self, max_bv_num: int, cfg_sumo='config/lane.sumocfg', output_path='', gui=False, seed=42) -> None:
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
        
        self.bv_num = 0                         # initialize with 0
        self.bv_pos = np.zeros((0, 2))          # initialize with 0
        self.bv_vel = np.zeros((0, 2))          # initialize with 0
        self.total_timestep = 0                 # initialize with 0
        self.road_len = 0                       # initialize with 0
        self.max_bv_num = max_bv_num
        self.state_dim = (max_bv_num + 1) * 4   # x_pos, y_pos, speed, yaw
        self.action_dim = 2                     # delta_speed, delta_yaw
        self.action_range = ((-6, 2), (-3, 3))  # the unit of angle is degree
        self.current_episode = 0
        
        command = [checkBinary(app), "--start", '-c', self.cfg_sumo]
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
        command += ['--tripinfo-output',self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        
        traci.start(command)
    
    def reset(self, scenario: np.ndarray, av_speed: float):
        # clear all vehicles
        for vehicle in traci.vehicle.getIDList():
            traci.vehicle.remove(vehicle)

        scenario = scenario.reshape((-1, 5))
        self.total_timestep = int(np.max(scenario[:, 0]))
        self.bv_num = int(np.max(scenario[:, 1]))
        self.road_len = math.ceil(np.max(scenario[:, 2]))
        self.bv_pos = np.zeros((self.bv_num, 2))
        self.bv_vel = np.zeros((self.bv_num, 2))
        
        cur_time = float(traci.simulation.getTime())
        
        # add autonomous vehicle and move it to its initial position
        AVID = "AV.%d" % self.current_episode
        traci.vehicle.add(vehID=AVID, routeID='straight', typeID='AV', depart=cur_time, departSpeed=av_speed)
        traci.vehicle.moveToXY(vehID=AVID, edgeID='', lane=0, x=scenario[0,2], y=scenario[0,3], 
                               angle=scenario[0,4], matchThreshold=self.road_len)
        traci.vehicle.setLaneChangeMode(AVID, 0b000000000000)
        traci.vehicle.setSpeedMode(AVID, 0b100000)
        
        # add background vehicles and move them to its initial position
        for i in range(self.bv_num):
            BVID = "BV.%d" % (i+1)
            traci.vehicle.add(vehID=BVID, routeID='straight', typeID='BV', depart=cur_time)
            traci.vehicle.moveToXY(vehID=BVID, edgeID='', lane=0, x=scenario[i+1,2], y=scenario[i+1,3], 
                                   angle=scenario[i+1,4], matchThreshold=self.road_len)
            traci.vehicle.setLaneChangeMode(BVID, 0b000000000000)
            traci.vehicle.setSpeedMode(BVID, 0b100000)
            self.bv_pos[i,0] = traci.vehicle.getPosition(BVID)[0]
            self.bv_pos[i,1] = traci.vehicle.getPosition(BVID)[1]
            self.bv_vel[i,0] = traci.vehicle.getSpeed(BVID)
            self.bv_vel[i,1] = traci.vehicle.getAngle(BVID)
        
        traci.simulationStep()
        while AVID not in traci.simulation.getDepartedIDList():
            traci.simulationStep()
        
        self.current_episode += 1

        return self.get_state(AVID)
    
    def get_state(self, AVID):
        pass
    
    def step(self, action: np.ndarray):
        pass
