from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs.onell_env import OneLLEnv, OneLLEnvDiscreteLbd, OneLLEnvRelativeControlLbd, OneLLEnvRelativeControlLbdDiscrete, RLSEnv, RLSEnvDiscreteK

import numpy as np
import os
import pandas as pd
import gym

ONELL_DEFAULTS = {
    "env_class": "OneLLEnv",
    "action_space_class": "Box",
    "action_space_args": [np.array([1]), np.array([100])],
    "action_description": "lbd",
    "normalise_lbd": False,
    "observation_description": "n, f(x)",
    "reward_range": [-np.inf, np.inf],   # the true reward range is instance dependent
    "reward_choice": "imp_minus_evals_normalised",
    "cutoff": 1e9,  # we don't really use this, 
                    # the real cutoff is in instance_set and is instance dependent
    "include_xprime": True, # if True, xprime is included in the selection after crossover phase
    "count_different_inds_only": True, # if True, only count an evaluation of a child if it is different from both of its parents
    "seed": 0,        
    "problem": "OneMax",
    "instance_set_path": "onemax_2000",
    "benchmark_info": "",
    "name": "OneLL"
}

class OneLLBenchmark(AbstractBenchmark):
    """
    Benchmark with various settings for (1+(lbd, lbd))-GA and RLS
    """

    def __init__(self, config=None):
        """
        Initialize OneLL benchmark

        Parameters            
        -------
        base_config_name: str
            OneLL's config name
            possible values: see ../additional_configs/onell/configs.py
        config : str
            a dictionary, all options specified in this argument will override the one in base_config_name

        """
        #if base_config_name is None:   
        #    base_config_name = 'lbd_theory'
        #config_path = os.path.dirname(os.path.abspath(__file__)) + '/../additional_configs/onell/' + base_config_name + '.json'        
        super(OneLLBenchmark, self).__init__()        

        self.config = objdict(ONELL_DEFAULTS)

        if config:
            for key, val in config.items():
                self.config[key] = val        

        self.read_instance_set()

        assert self.config.env_class in ['OneLLEnv','OneLLEnvDiscreteLbd','OneLLEnvRelativeControlLbd', 'OneLLEnvRelativeControlLbdDiscrete','RLSEnv','RLSEnvDiscreteK']
        
        self.env_class = globals()[self.config.env_class]

        # create observation space
        self.config['observation_space'] = self.create_observation_space_from_description(self.config['observation_description'], self.env_class)

        # initialise action space        
        if "Discrete" in config['env_class']:
            assert "action_choices" in config, "ERROR: action_choices is required for " + env_class
            n_acts = len(config['action_choices'])
            self.config['action_space'] = gym.spaces.Discrete(n_acts) 
        else:            
            # TODO: this only works for 1-dim action space
            assert "min_action" in config
            assert "max_action" in config
            self.config['action_space'] = gym.spaces.Box(low=np.array(config['min_action']), high=np.array(config['max_action']))


    def create_observation_space_from_description(self, obs_description, env_class=OneLLEnv):
        """
        Create a gym observation space (Box only) based on a string containing observation variable names, e.g. "n, f(x), lbd, lbd_{t-1}" (for onell)
        Return:
            A gym.spaces.Box observation space
        """
        obs_var_names = [s.strip() for s in obs_description.split(',')]
        low = []
        high = []
        for var_name in obs_var_names:
            l, h = env_class.get_obs_domain_from_name(var_name)
            low.append(l)
            high.append(h)
        obs_space = gym.spaces.Box(low=np.array(low), high=np.array(high))
        return obs_space


    def get_environment(self, test_env=False):
        """
        Return an environment with current configuration        
        """        
            
        env = self.env_class(self.config, test_env)
        
        for func in self.wrap_funcs:
            env = func(env)

        return env

    
    def read_instance_set(self):
        """Read instance set from file"""        
        assert self.config.instance_set_path        
        if os.path.isfile(self.config.instance_set_path):
            path = self.config.instance_set_path
        else:        
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/../instance_sets/onell/"
                + self.config.instance_set_path + ".csv"
            )                

        self.config["instance_set"] = pd.read_csv(path,index_col=0).to_dict('id')

        for key, val in self.config['instance_set'].items():
            self.config['instance_set'][key] = objdict(val)
