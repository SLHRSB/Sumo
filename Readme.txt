This is the python file that contains the Sumo environment class for GYM: HW Project\envs\sumo_env_dir\Sumo_env.py
After editing this file, you should re-install the environment. To do this, open Anaconda-cmd:

>>>cd HW Project_Github
>>>pip install -e .
----------------------------------------------------------------

To create an environment you need to make a runner.py file (next to the sumo files for a specific project), and have these:

import gym
import envs
env = gym.make('SumoEnv-v0')

----------------------------------------------------------------
in the highway_1.sumocfg file, the addreses for two highway_1.net.xml and highway_1.rou.xml files should be replaced properly.

----------------------------------------------------------------
To train the agant of the highway project, open Anaconda-cmd:
>>>cd Sumo HW Project_Github\Best_HW
>>>python main_RL.py

the sumo config file will be oppen and you should run it to start gathering data. 