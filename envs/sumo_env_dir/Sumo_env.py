from gym import Env
import os
import sys
import optparse
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import traci
import subprocess
import traci.constants as tc
import math
import time

# for SAC project
class SumoEnv(Env):

	def __init__(self, mode='gui', simulation_end=36000):
		print("-------------------------------- Best Environment created!!! --------------------------------")
		self.simulation_end = simulation_end
		self.mode = mode
		self.seed()
		self.traci = self.initSimulator(True, 8870)
		self.sumo_step = 0
		self.collisions = 0
		self.long = 0
		self.episode = 0
		# self.flag = True
		self.terminalType = None
		self.car_not_found = 0

		## INITIALIZE EGO CAR
		self.egoCarID = "EGO" # 'veh0'
		self.speed = 0
		self.max_speed = 20.1168  # m/s

		self.sumo_running = False
		self.viewer = None
		self.state = None

		high = np.array([150, 3.5, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
		self.action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32)  # acceleration  #self.action_space = spaces.Box(np.array([-1,-2]),np.array([2,4]),dtype=np.float32)
		self.observation_space = spaces.Box(np.array([0,0,0,0]), np.array([1,1,1,1]), dtype=np.float32)

	def reset(self):
		try:
			# self.traci.vehicle.remove("LeaderCar")
			self.traci.vehicle.remove(self.egoCarID)
			# print ("---leader removed")
		except:
			pass

		print(" ")
		# self.addLeader()
		self.addEgoCar()  # Add the ego car to the scene
		self.setGoalPosition()  # Set the goal position

		self.speed = 0
		dt = self.traci.simulation.getDeltaT() / 1000.0
		self.traci.vehicle.slowDown(self.egoCarID, self.speed, int(dt * 1000))

		self.sumo_step = 0
		self.traci.simulationStep()  # Take a simulation step to initialize car
		self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
		# self.state = self._observation()
		return np.array(self.state)

	def step(self, action):

		# print ("speed: ", self.speed)
		reward_step = 0
		self.sumo_step += 1
		# print("action= ", action)
		self.takeAction(action)
		# print("speed = ", self.speed)
		self.traci.simulationStep()
		# Get reward and check for terminal state
		terminal, self.terminalType = self.terminalCheck()
		reward = self.reward(terminal, self.terminalType)
		reward_step += reward
		if not terminal:
			self.state = self.observation()

		return np.array(self.state), reward_step, terminal, {}

	def observation(self):

		try:
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
		except:
			print("--Traci couldn't find car, observation")
			self.speed = 0
			self.reset()
			return  True, 'Car not found'

		ego_x, ego_y = self.traci.vehicle.getPosition(self.egoCarID)
		self.speed = self.traci.vehicle.getSpeed(self.egoCarID)
		ego_a = self.traci.vehicle.getAcceleration(self.egoCarID)

		#leader:
		leader_dis = 100
		leader_speed = 30
		try:
			leader = self.traci.vehicle.getLeader(self.egoCarID)
			if leader[0] != None: # leader[0] = leader ID
				leader_dis = leader[1]
				leader_speed = self.traci.vehicle.getSpeed(leader[0])
		except:
			leader_dis = 100
			leader_speed = 30
			# print("no leader")

		# Traffic
		# for carID in self.traci.vehicle.getIDList():
		# 	if carID == self.egoCarID:
		# 		continue
		# 	c_x, c_y = self.traci.vehicle.getPosition(carID)
		# 	c_v = self.traci.vehicle.getSpeed(carID)
		# 	c_a = self.traci.vehicle.getAcceleration(carID)
		# 	p_x = c_x - ego_x
		# 	p_y = c_y - ego_y

		# return (ego_x, ego_y, ego_v, ego_a/5)
		# return (ego_x/100, self.speed/30, self.sumo_step/100, leader_dis/100 )

		return ( self.speed / 30, leader_dis / 100, leader_speed/30,  self.sumo_step / 100)

	def reward(self, terminal, terminalType):

		# Step cost
		reward = -0.05

		# Speed Reward
		# reward += int(self.speed)/30
		# if self.speed != 0:
		#     reward += 0.04
		max_reward_speed = 1
		if int(self.speed) <= 25:
			reward_speed = max_reward_speed * int(self.speed) / 30
		else:
			reward_speed = -max_reward_speed * int(self.speed) / 30
		reward += reward_speed

		# Cooperation Reward
		# traffic_waiting = self.isTrafficWaiting()
		# traffic_braking = self.isTrafficBraking()
		#
		# if (traffic_waiting and traffic_braking):
		# 	reward += -0.05
		# elif (traffic_braking or traffic_waiting):
		# 	reward += -0.025
		# elif (not traffic_waiting and not traffic_braking):
		# 	reward += + 0.05

		if terminal:
			if terminalType == 'Collided':
				reward += -10.0
			elif terminalType ==  'Survived':
				reward += 5.0
			elif terminalType == 'Took too long':
				reward += -10.0
			elif terminalType == "car not found":
				reward += 0

		# print("Reward = ", reward)
		return reward

	def terminalCheck(self):

		terminal = 0
		self.terminalType = None

		try:
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
		except:
			print("--Traci couldn't find car, terminal check")
			terminal = True
			self.terminalType =  'Car not found'
			print("terminalType:", self.terminalType)
			self.car_not_found +=1
			self.reset()
			return terminal, self.terminalType

		# Collision check
		leader = self.traci.vehicle.getLeader(self.egoCarID)
		# print("leader id and distance: " , leader)
		if leader != None and leader[1] <= 1:
			self.collisions += 1
			terminal = 1
			self.terminalType = 'Collided'
			self.episode += 1
			print("terminalType:", self.terminalType, "at position: ", position_ego[0])
			# if (self.episode % 100 == 0):
			# 	self.flag = True


		# end road check
		position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))

		if position_ego[0] >= self.endPos[0]:
			terminal = 1
			self.terminalType = 'Survived'
			self.episode += 1
			print("terminalType:", self.terminalType)
			# if (self.episode % 100 == 0):
			# 	self.flag = True

		# Taking long time to complete the episode
		if self.sumo_step >= 150 and self.speed <0.1  : #and position_ego[0]<20
			self.long += 1
			terminal = 1
			self.terminalType = 'Took too long'
			self.episode += 1
			print("terminalType:", self.terminalType)
			# if (self.episode % 100 == 0):
			# 	self.flag = True

		return terminal, self.terminalType

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def initSimulator(self, withGUI, portnum):

		if 'SUMO_HOME' in os.environ:
			tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
			sys.path.append(tools)
		else:
			sys.exit("please declare environment variable 'SUMO_HOME'")

		from sumolib import checkBinary  # Checks for the binary in environ vars
		import traci

		def get_options():
			opt_parser = optparse.OptionParser()
			opt_parser.add_option("--nogui", action="store_true",
								  default=False, help="run the commandline version of sumo")
			options, args = opt_parser.parse_args()
			return options

		options = get_options()
		if options.nogui:
			sumoBinary = checkBinary('sumo')
		else:
			sumoBinary = checkBinary('sumo-gui')

		sumoConfig = "highway_1.sumocfg"

		# traci starts sumo as a subprocess and then this script connects and runs
		# traci.start([sumoBinary, "-c", "3.sumocfg", "--tripinfo-output", "tripinfo.xml"])

		# Call the sumo simulator
		sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
										"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
										"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout,
									   stderr=sys.stderr)

		# Initialize the simulation
		traci.init(portnum)
		return traci

	def closeSimulator(traci):
		traci.close()
		sys.stdout.flush()

	def addEgoCar(self):

		vehicles = self.traci.vehicle.getIDList()

		## PRUNE IF TRAFFIC HAS BUILT UP TOO MUCH
		# if more cars than setnum, p(keep) = setnum/total
		setnum = 20
		if len(vehicles) > 0:
			keep_frac = float(setnum) / len(vehicles)
		for i in range(len(vehicles)):
			if vehicles[i] != self.egoCarID:
				if np.random.uniform(0, 1, 1) > keep_frac:
					self.traci.vehicle.remove(vehicles[i])

		## DELAY ALLOWS CARS TO DISTRIBUTE
		for j in range(np.random.randint(5,8)):  # np.random.randint(0,10)):
			self.traci.simulationStep()

		## STARTING LOCATION
		# depart = -1   (immediate departure time)
		# pos    = -2   (random position)
		# speed  = -2   (random speed)
		self.traci.vehicle.addFull(self.egoCarID, 'routeEgo', depart=None, departPos='0.0', departSpeed='0',
								   departLane='0', typeID='vType0')
		# self.traci.vehicle.addFull("ego2", 'routeEgo', depart=None, departPos='0.0', departSpeed='0',
		# 						   departLane='0', typeID='vType2')
		print ("ego added")

		self.traci.vehicle.setSpeedMode(self.egoCarID, int('00000', 2))


	def addLeader(self):

		## DELAY ALLOWS CARS TO DISTRIBUTE
		# for j in range(np.random.randint(40, 50)):  # np.random.randint(0,10)):
		# 	self.traci.simulationStep()
		self.traci.vehicle.addFull("LeaderCar", 'routeEgo', depart=None, departPos='5.0', departSpeed='0',
								   departLane='0', typeID='vType2')
		print ("Leader added")

		self.traci.vehicle.setSpeedMode("LeaderCar", int('00000', 2))

	def setGoalPosition(self):
		self.endPos = [110, 0]

	def takeAction(self, action):

		dt = self.traci.simulation.getDeltaT() / 1000.0
		self.speed = self.speed + action

		if  self.speed <= 0:
			self.speed = 0

		self.traci.vehicle.slowDown(self.egoCarID, self.speed, int(dt * 1000))

	def isTrafficBraking(self):
		""" Check if any car is braking
		"""
		for carID in self.traci.vehicle.getIDList():
			if carID != self.egoCarID:
				brakingState = self.traci.vehicle.getSignals(carID)
				if brakingState == 8:
					return True
		return False

	def isTrafficWaiting(self):
		""" Check if any car is waiting
		"""
		for carID in self.traci.vehicle.getIDList():
			if carID != self.egoCarID:
				speed = self.traci.vehicle.getSpeed(carID)
				if speed <= 1e-1:
					return True
		return False

	def getBinnedFeature(self, val, start, stop, numBins):
		""" Creating binary features.
		"""
		bins = np.linspace(start, stop, numBins)
		binaryFeatures = np.zeros(numBins)

		if val == 'unknown':
			return binaryFeatures

		# Check extremes
		if val <= bins[0]:
			binaryFeatures[0] = 1
		elif val > bins[-1]:
			binaryFeatures[-1] = 1

		# Check intermediate values
		for i in range(len(bins) - 1):
			if val > bins[i] and val <= bins[i + 1]:
				binaryFeatures[i + 1] = 1

		return binaryFeatures

def wrapPi(angle):
	# makes a number -pi to pi
	while angle <= -180:
		angle += 360
	while angle > 180:
		angle -= 360
	return angle