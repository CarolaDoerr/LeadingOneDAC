import numpy as np
from copy import deepcopy
import logging
from collections import deque

import sys
import os
import uuid
import gym

from dacbench import AbstractEnv

class BinaryProblem:
    """
    An abstract class for an individual in binary representation
    """
    def __init__(self, n, val=None, rng=np.random.default_rng()):
        if val is not None:
            assert isinstance(val, bool)
            self.data = np.array([val] * n)
        else:
            self.data = rng.choice([True,False], size=n) 
        self.n = n
        self.fitness = self.eval()

    
    def initialise_with_fixed_number_of_bits(self, k, rng=np.random.default_rng()):
        nbits = self.data.sum()        
        if nbits < k:            
            ids = rng.choice(np.where(self.data==False)[0], size=k-nbits, replace=False)
            self.data[ids] = True
            self.eval()
        

    def is_optimal(self):
        pass


    def get_optimal(self):
        pass


    def eval(self):
        pass        


    def get_fitness_after_flipping(self, locs):
        """
        Calculate the change in fitness after flipping the bits at positions locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: int
        -----------
            objective after flipping
        """
        raise NotImplementedError

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """
        Calculate fitness of the child aftering being crossovered with xprime

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: fitness of the new individual after crossover
        -----------            
        """
        raise NotImplementedError

    def flip(self, locs):
        """
        flip the bits at position indicated by locs

        Parameters
        -----------
            locs: 1d-array
                positions where bits are flipped

        Returns: the new individual after the flip
        """
        child = deepcopy(self)
        child.data[locs] = ~child.data[locs]
        child.eval()
        return child

    def combine(self, xprime, locs_xprime):
        """
        combine (crossover) self and xprime by taking xprime's bits at locs_xprime and self's bits at other positions

        Parameters
        -----------
            xprime: 1d boolean array
                the individual to crossover with
            locs_x: 1d boolean/integer array
                positions where we keep current bits of self
            locs_xprime: : 1d boolean/integer array
                positions where we change to xprime's bits

        Returns: the new individual after the crossover        

        """
        child = deepcopy(self)
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.eval()
        return child

    def mutate(self, p, n_childs, rng=np.random.default_rng()):
        """
        Draw l ~ binomial(n, p), l>0
        Generate n_childs children by flipping exactly l bits
        Return: the best child (maximum fitness), its fitness and number of evaluations used        
        """
        assert p>=0

        if p==0:
            return self, self.fitness, 0

        l = 0
        while l==0:
            l = rng.binomial(self.n, p)                
        
        best_obj = -1
        best_locs = None
        for i in range(n_childs):
            locs = rng.choice(self.n, size=l, replace=False)        
            obj = self.get_fitness_after_flipping(locs)
            if obj > best_obj:
                best_locs = locs
                best_obj = obj                       

        best_child = self.flip(best_locs)                

        return best_child, best_child.fitness, n_childs

    def mutate_rls(self, l, rng=np.random.default_rng()):
        """
        generate a child by flipping exactly l bits
        Return: child, its fitness        
        """
        assert l>=0

        if l==0:
            return self, self.fitness, 0

        locs = rng.choice(self.n, size=l, replace=False) 
        child = self.flip(locs)

        return child, child.fitness, 1   

    def crossover(self, xprime, p, n_childs, 
                    include_xprime=True, count_different_inds_only=True,
                    rng=np.random.default_rng()):
        """
        Crossover operator:
            for each bit, taking value from x with probability p and from self with probability 1-p
        Arguments:
            x: the individual to crossover with
            p (float): in [0,1]                                                
        """
        assert p <= 1
        
        if p == 0:
            if include_xprime:
                return xprime, xprime.fitness, 0
            else:
                return self, self.fitness, 0            

        if include_xprime:
            best_obj = xprime.fitness
        else:
            best_obj = -1            
        best_locs = None

        n_evals = 0
        ls = rng.binomial(self.n, p, size=n_childs)        
        for l in ls:                   
            locs_xprime = rng.choice(self.n, l, replace=False)
            locs_x = np.full(self.n, True)
            locs_x[locs_xprime] = False
            obj = self.get_fitness_after_crossover(xprime, locs_x, locs_xprime) 
                   
            if (obj != self.fitness) and (obj!=xprime.fitness):
                n_evals += 1
            elif (not np.array_equal(xprime.data[locs_xprime], self.data[locs_xprime])) and (not np.array_equal(self.data[locs_x], xprime.data[locs_x])):            
                n_evals += 1            

            if obj > best_obj:
                best_obj = obj
                best_locs = locs_xprime
            
            
        if best_locs is not None:
            child = self.combine(xprime, best_locs)
        else:
            child = xprime

        if not count_different_inds_only:
            n_evals = n_childs

        return child, child.fitness, n_evals


class OneMax(BinaryProblem):
    """
    An individual for OneMax problem
    The aim is to maximise the number of 1 bits
    """

    def eval(self):
        self.fitness = self.data.sum()
        return self.fitness

    def is_optimal(self):
        return self.data.all()

    def get_optimal(self):
        return self.n

    def get_fitness_after_flipping(self, locs):        
        # f(x_new) = f(x) + l - 2 * sum_of_flipped_block
        return self.fitness + len(locs) - 2 * self.data[locs].sum()

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):        
        return self.data[locs_x].sum() + xprime.data[locs_xprime].sum()
        

class LeadingOne(BinaryProblem):    
    """
    An individual for LeadingOne problem
    The aim is to maximise the number of leading (and consecutive) 1 bits in the string
    """

    def eval(self):
        k = self.data.argmin()
        if self.data[k]:
            self.fitness = self.n
        else:
            self.fitness = k
        return self.fitness

    def is_optimal(self):
        return self.data.all()  

    def get_optimal(self):
        return self.n    

    def get_fitness_after_flipping(self, locs):        
        min_loc = locs.min()
        if min_loc < self.fitness:
            return min_loc
        elif min_loc > self.fitness:
            return self.fitness
        else:
            old_fitness = self.fitness
            self.data[locs] = ~self.data[locs]
            new_fitness = self.eval()            
            self.data[locs] = ~self.data[locs]
            self.fitness = old_fitness
            return new_fitness


    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        child = self.combine(xprime, locs_xprime)                
        child.eval()
        return child.fitness
        

MAX_INT = 1e8
HISTORY_LENGTH = 5

class OneLLEnv(AbstractEnv):
    """
    Environment for (1+(lbd, lbd))-GA
    for both OneMax and LeadingOne problems
    """

    def __init__(self, config, test_env=False) -> None:
        """
        Initialize OneLLEnv

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(OneLLEnv, self).__init__(config)        
        self.logger = logging.getLogger(self.__str__())

        self.test_env = test_env

        self.name = config.name   

        # whether we start at an inital solution with fixed objective value or not
        #   if config.init_solution_ratio is not None, we start at a solution with f = n * init_solution_ratio
        self.init_solution_ratio = None
        if ('init_solution_ratio' in config) and (config.init_solution_ratio!=None) and (config.init_solution_ratio!='None'):            
            self.init_solution_ratio = float(config.init_solution_ratio)   
            self.logger.info("Starting from initial solution with f = %.2f * n" % (self.init_solution_ratio))     

        # name of reward function
        assert config.reward_choice in ['imp_div_evals', 'imp_div_evals_new', 'imp_minus_evals', 'minus_evals', 'imp', 'minus_evals_normalised', 'imp_minus_evals_normalised', 'imp_minus_normalised_evals']
        self.reward_choice = config.reward_choice
        #print("Reward choice: " + self.reward_choice)        
        
        # OneLL-GA's setting
        self.problem = globals()[config.problem]
        self.include_xprime = config.include_xprime
        self.count_different_inds_only = config.count_different_inds_only
      
        # read names of all observation variables
        self.obs_description = config.observation_description
        self.obs_var_names = [s.strip() for s in config.observation_description.split(',')]

        # functions to get values of the current state from histories 
        # (see reset() function for those history variables)        
        self.state_functions = []
        for var_name in self.obs_var_names:            
            state_func = self.get_state_function(var_name)
            if state_func is None:
                raise Exception("Error: invalid state variable name: " + var_name)
            self.state_functions.append(state_func)
        
        # names of all variables in an action        
        self.action_description = config.action_description
        self.action_var_names = [s.strip() for s in config.action_description.split(',')] # names of 
        for name in self.action_var_names:
            assert name in self.get_valid_action_var_names(), "Error: invalid action variable name: " + name

        # the random generator used by OneLL-GA
        if 'seed' in config:
            seed = config.seed
        else:
            seed = None
        self.rng = np.random.default_rng(seed)   

        # for logging
        self.outdir = None
        if 'outdir' in config:
            self.outdir = config.outdir + '/' + str(uuid.uuid4())
            #self.log_fn_rew_per_state        

    def get_valid_action_var_names(self):
        return ['lbd','lbd_crossover','p','c']

    def get_obs_domain_from_name(var_name):
        """
        Get default lower and upperbound of a observation variable based on its name.
        The observation space will then be created 
        Return:
            Two int values, e.g., 1, np.inf
        """
        if (var_name[0]=='p') or (var_name[0]=='c'):
            return 0, 1
        return 1, np.inf            

    def get_state_function(self, var_name):
        """
        Get a function that return a component (var_name) of the current state

        Returns
        -------
            A function
        """
        if var_name == 'n': # current problem size
            return lambda: self.n
        if var_name in ['lbd','lbd_crossover', 'p', 'c']: # current onell params 
            return lambda his='history_'+var_name: vars(self)[his][-1]
        if "_{t-" in var_name: # onell params in previous iterations, e.g., lbd1_{t-1}
            k = int(var_name.split("_{t-")[1][:-1]) # get the number in _{t-<number>}
            name = var_name.split("_{t-")[0] # get the variable name (lbd, lbd1, etc)
            return lambda his='history_'+name: vars(self)[his][-(k+1)] # the last element is the value at the current time step, so we have to go one step back to access the history
        if var_name == "f(x)":
            return lambda: self.history_fx[-1]
        if var_name == "delta_f(x)":
            return lambda: self.history_fx[-1] - self.history_fx[-2]
        elif var_name == "optimal_lbd_theory":
            return lambda: np.sqrt(self.n/max(1,self.n-self.history_fx[-1]))
        return None

    def seed(self, seed=None, seed_action_space=False):
        super(OneLLEnv, self).seed(seed, seed_action_space)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """        
        super(OneLLEnv, self).reset_()        

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        self.max_evals = self.instance.max_evals
        self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

        # create an initial solution
        self.x = self.problem(n=self.instance.size, rng=self.rng)
        if self.init_solution_ratio:
            self.x.initialise_with_fixed_number_of_bits(int(self.init_solution_ratio * self.x.n))

        # total number of evaluations so far
        self.total_evals = 1                

        # reset histories
        self.history_lbd = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_lbd_crossover = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_p = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_c = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_fx = deque([self.x.fitness]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

        # for logging
        self.log_lbd = [] 
        self.log_lbd_crossover = []
        self.log_fx = []
        self.log_p = []
        self.log_c = []
        self.log_reward = []    
        self.log_eval = []
        self.init_obj = self.x.fitness 
        
        return self.get_state()


    def get_state(self):
        return np.asarray([f() for f in self.state_functions])

    def get_onell_lbd(self, action):
        return action[self.action_var_names.index('lbd')]

    def get_onell_lbd_crossover(self, action, lbd):
        if 'lbd_crossover' in self.action_var_names:
            return action[self.action_var_name.index('lbd_crossover')]
        return lbd

    def get_onell_p(self, action, lbd):
        if 'p' in self.action_var_names:
            return action[self.action_var_name.index('p')]
        return lbd/self.n

    def get_onell_c(self, action, lbd):
        if 'c' in self.action_var_names:
            return action[self.action_var_name.index('c')]
        if lbd==0:
            return np.inf
        return 1/lbd 
 
    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------            
            state, reward, done, info
            np.array, float, bool, dict
        """
        super(OneLLEnv, self).step_()                
                
        fitness_before_update = self.x.fitness

        # convert action to list
        if (not isinstance(action, np.ndarray)) and (not isinstance(action, list)):
            action = [action]

        lbd = self.get_onell_lbd(action)
        lbd_crossover, p, c = self.get_onell_lbd_crossover(action,lbd), self.get_onell_p(action,lbd), self.get_onell_c(action,lbd)

        # if onell params are out of range, return a large negative reward and stop the episode
        stop = False
        if lbd<1 or lbd>self.n or lbd_crossover<1 or lbd_crossover>self.n or p<0 or p>1 or c<0 or c>1:
            self.logger.info("WARNING: action is out of bound (%.2f, %.2f, %.2f, %.2f)" % (lbd, lbd_crossover, p, c))
            if self.test_env is False:
                done = True
                n_evals = 0
                reward = -MAX_INT
                stop = True
            else:
                lbd = np.clip(lbd, 1, self.n)
                lbd_crossover, p, c = self.get_onell_lbd_crossover(action,lbd), self.get_onell_p(action,lbd), self.get_onell_c(action,lbd)
                lbd_crossover = np.clip(lbd_crossover, 1, self.n)
                p = np.clip(p, 0, 1)
                c = np.clip(c, 0, 1)
                #print("lbd=%f; p=%f; c=%f" % (lbd, p, c))

        if stop is False:
            # mutation phase
            xprime, f_xprime, ne1 = self.x.mutate(p, int(lbd), self.rng)

            # crossover phase
            y, f_y, ne2 = self.x.crossover(xprime, c, int(lbd_crossover), self.include_xprime, self.count_different_inds_only, self.rng)        
            
            # update x
            if self.x.fitness <= y.fitness:
                self.x = y
            
            # update total number of evaluations
            n_evals = ne1 + ne2
            self.total_evals += n_evals

            # check stopping criteria        
            done = (self.total_evals>=self.instance.max_evals) or (self.x.is_optimal())        
            
            # calculate reward        
            imp = self.x.fitness - fitness_before_update
            if self.reward_choice=='imp_div_evals':        
                reward = imp / n_evals
            elif self.reward_choice=='imp_div_evals_new':            
                reward = (self.x.fitness - fitness_before_update - 0.5) / n_evals
            elif self.reward_choice=='imp_minus_evals':
                reward = self.x.fitness - fitness_before_update - n_evals
            elif self.reward_choice=='minus_evals':
                reward = -n_evals
            elif self.reward_choice=='minus_evals_normalised':
                reward = -n_evals / self.max_evals            
            elif self.reward_choice=='imp_minus_evals_normalised':
                reward = (self.x.fitness - fitness_before_update - n_evals) / self.max_evals
            elif self.reward_choice=='imp_minus_normalised_evals':
                reward = self.x.fitness - fitness_before_update - n_evals/self.max_evals
            elif self.reward_choice=='imp':
                reward = self.x.fitness - fitness_before_update 

        # update histories
        self.history_lbd.append(lbd)
        self.history_lbd_crossover.append(lbd_crossover)
        self.history_p.append(p)
        self.history_c.append(c)
        self.history_fx.append(self.x.fitness)

        # update logs
        self.log_fx.append(int(self.x.fitness))
        self.log_lbd.append(float(lbd))
        self.log_lbd_crossover.append(float(lbd_crossover))
        self.log_p.append(float(p))
        self.log_c.append(float(c))
        self.log_reward.append(float(reward))
        self.log_eval.append(int(n_evals))
        
        returned_info = {"msg": "", "values":{}}
        if done:
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""    
            msg += "Episode done: n=%d; obj=%d; init_obj=%d; evals=%d; max_evals=%d; steps=%d; lbd_min=%.1f; lbd_max=%.1f; lbd_mean=%.1f; R=%.4f" % (self.n, self.x.fitness, self.init_obj, self.total_evals, self.max_evals, self.c_step, min(self.log_lbd), max(self.log_lbd), sum(self.log_lbd)/len(self.log_lbd), sum(self.log_reward))      
            #self.logger.info(msg) 
            returned_info['msg'] = msg
            returned_info['values'] = {'n':int(self.n), 
                                        'obj': int(self.x.fitness), 
                                        'init_obj': int(self.init_obj), 
                                        'evals': int(self.total_evals), 
                                        'max_evals': int(self.max_evals), 
                                        'steps': int(self.c_step), 
                                        'lbd_min': float(min(self.log_lbd)), 
                                        'lbd_max': float(max(self.log_lbd)), 
                                        'lbd_mean': float(sum(self.log_lbd)/len(self.log_lbd)), 
                                        'R': float(sum(self.log_reward)), 
                                        'log_lbd': self.log_lbd,
                                        'log_lbd_crossover': self.log_lbd_crossover,
                                        'log_p': self.log_p,
                                        'log_c': self.log_c,
                                        'log_reward': self.log_reward,
                                        'log_evals': self.log_eval,
                                        'log_fx': self.log_fx}
        
        return self.get_state(), reward, done, returned_info
            

    def close(self) -> bool:
        """
        Close Env

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation
        """        
        return True


class OneLLEnvDiscreteLbd(OneLLEnv):
    """
    OneLL environment where lbd choice is discretised
    If the chosen lbd is out of range:
        - if config['clip_lbd']=False: a reward of -MAX_INT will be returned and the episode is terminated (default OneLLEnv behaviour)
        - otherwise: lbd is clipped into [1,n], and a reward of -MAX_INT will be returned
    """
    def __init__(self, config, test_env=False) -> None:
        assert 'action_choices' in config, "Error: action_choices must be specified in benchmark's config"
        assert 'clip_lbd' in config, "Error: clip_lbd must be specified in benchmark's config"
        super(OneLLEnvDiscreteLbd, self).__init__(config, test_env)
        assert isinstance(self.action_space, gym.spaces.Discrete), "Error: action space must be discrete"
        assert self.action_space.n == len(config['action_choices']), "Error: action space's size (%d) must be equal to the len(action_choices) (%d)" % (self.action_space.n, len(config['action_choices']))
        self.action_choices = config['action_choices']
        self.clip_lbd = config['clip_lbd']

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action)==1
            action = action[0]
        lbd = self.action_choices[action]
        if (self.test_env is False) and self.clip_lbd and (lbd<1 or lbd>self.n):
            lbd = np.clip(lbd, 1, self.n)
            s, r, d, info = super(OneLLEnvDiscreteLbd, self).step([lbd])
            return s, -MAX_INT, d, info
        else:
            return super(OneLLEnvDiscreteLbd, self).step([lbd])


class OneLLEnvRelativeControlLbd(OneLLEnv):
    """
    OneLL environment where lbd is controlled relatively (lbd = lbd_{t-1} * action)
    """
    def __init__(self, config, test_env) -> None:
        assert "clip_lbd" in config, "Error: clip_lbd must be specified in benchmark's config"
        super(OneLLEnvRelativeControlLbd, self).__init__(config, test_env)
        self.clip_lbd = config['clip_lbd']

    def reset(self):
        self.prev_lbd = 1
        return super(OneLLEnvRelativeControlLbd, self).reset()

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action)==1
            action = action[0]
        lbd = self.prev_lbd * action
        if (self.test_env is False) and self.clip_lbd and (lbd<1 or lbd>self.n):
            lbd = np.clip(lbd, 1, self.n)
            s, r, d, info = super(OneLLEnvRelativeControlLbd, self).step([lbd])
            self.prev_lbd = lbd
            return s, -MAX_INT, d, info
        else:
            self.prev_lbd = lbd
            return super(OneLLEnvRelativeControlLbd, self).step([lbd])


class OneLLEnvRelativeControlLbdDiscrete(OneLLEnv):
    """
    OneLL environment where lbd is controlled relatively (lbd = lbd_{t-1} * action)
    """
    def __init__(self, config, test_env=False) -> None:
        assert "clip_lbd" in config, "Error: clip_lbd must be specified in benchmark's config"
        assert "action_choices" in config, "Error: action_choices must be specified in benchmark's config"
        super(OneLLEnvRelativeControlLbdDiscrete, self).__init__(config, test_env)
        assert isinstance(self.action_space, gym.spaces.Discrete), "Error: action space must be discrete"
        assert self.action_space.n == len(config['action_choices']), "Error: action space's size (%d) must be equal to the len(action_choices) (%d)" % (self.action_space.n, len(config['action_choices']))
        self.clip_lbd = config['clip_lbd']
        self.action_choices = config['action_choices']

    def reset(self):
        self.prev_lbd = 1
        return super(OneLLEnvRelativeControlLbdDiscrete, self).reset()

    def step(self, action):
        if isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action)==1
            action = action[0]
        lbd = self.prev_lbd * self.action_choices[action] 
        if (self.test_env is False) and self.clip_lbd and (lbd<1 or lbd>self.n):
            lbd = np.clip(lbd, 1, self.n)
            s, r, d, info = super(OneLLEnvRelativeControlLbdDiscrete, self).step([lbd])
            self.prev_lbd = lbd
            return s, -MAX_INT, d, info
        else:
            self.prev_lbd = lbd
            return super(OneLLEnvRelativeControlLbdDiscrete, self).step([lbd])

class RLSEnv(AbstractEnv):
    """
    Environment for RLS with step size
    for both OneMax and LeadingOne problems
    Current assumption: we only consider (1+1)-RLS, so there's only one parameter to tune (k)
    """

    def __init__(self, config, test_env=False) -> None:
        """
        Initialize RLSEnv

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(RLSEnv, self).__init__(config)        
        self.logger = logging.getLogger(self.__str__())     

        self.test_env = test_env
        
        self.name = config.name   
        
        # name of reward function
        assert config.reward_choice in ['imp_div_evals', 'imp_div_evals_new', 'imp_minus_evals', 'minus_evals', 'imp', 'minus_evals_normalised', 'imp_minus_evals_normalised']
        self.reward_choice = config.reward_choice        
        #print("Reward choice: " + self.reward_choice)        

        # get problem
        self.problem = globals()[config.problem]                

        # read names of all observation variables
        self.obs_description = config.observation_description
        self.obs_var_names = [s.strip() for s in config.observation_description.split(',')]

        # functions to get values of the current state from histories 
        # (see reset() function for those history variables)        
        self.state_functions = []
        for var_name in self.obs_var_names:
            if var_name == 'n':
                self.state_functions.append(lambda: self.n)
            elif var_name in ['k']:
                self.state_functions.append(lambda his='history_'+var_name: vars(self)[his][-1])
            elif "_{t-" in var_name: # TODO: this implementation only allow accessing history of k, but not delta_f(x), optimal_k, etc
                k = int(var_name.split("_{t-")[1][:-1]) # get the number in _{t-<number>}
                name = var_name.split("_{t-")[0] # get the variable name (lbd, lbd1, etc)
                self.state_functions.append(lambda his='history_'+name: vars(self)[his][-(k+1)]) # the last element is the value at the current time step, so we have to go one step back to access the history
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta_f(x)":
                self.state_functions.append(lambda: self.history_fx[-1] - self.history_fx[-2])
            elif var_name == "optimal_k":
                self.state_functions.append(lambda: int(self.n/(self.history_fx[-1]+1)))
            else:
                raise Exception("Error: invalid state variable name: " + var_name)
        
        # the random generator used by RLS
        if 'seed' in config:
            seed = config.seed
        else:
            seed = None
        self.rng = np.random.default_rng(seed)   
        
        # for logging
        self.outdir = None
        if 'outdir' in config:
            self.outdir = config.outdir + '/' + str(uuid.uuid4())

    def seed(self, seed=None, seed_action_space=False):
        super(RLSEnv, self).seed(seed, seed_action_space)
        self.rng = np.random.default_rng(seed)
        
    def get_obs_domain_from_name(var_name):
        """
        Get default lower and upperbound of a observation variable based on its name.
        The observation space will then be created 
        Return:
            Two int values, e.g., 1, np.inf
        """        
        return 0, np.inf    
    
    def reset(self):
        """
        Resets env

        Returns
        -------
        numpy.array
            Environment state
        """        
        super(RLSEnv, self).reset_()        

        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        self.max_evals = self.instance.max_evals
        self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

        # create an initial solution
        self.x = self.problem(n=self.instance.size, rng=self.rng)

        # total number of evaluations so far
        self.total_evals = 1                        

        # reset histories
        self.history_k = deque([0]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)         
        self.history_fx = deque([self.x.fitness]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) 

        # for debug only
        self.log_k = []
        self.log_reward = []     
        self.log_fx = []
        self.init_obj = self.x.fitness 
        
        return self.get_state()
    
    def get_state(self):
        return np.asarray([f() for f in self.state_functions])

    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------            
            state, reward, done, info
            np.array, float, bool, dict
        """
        super(RLSEnv, self).step_()     

        fitness_before_update = self.x.fitness
        
        # get k
        if isinstance(action, np.ndarray) or isinstance(action, list):
            assert len(action)==1
            k = action[0]
        else:
            k = action   
            
        # if k is out of range
        stop = False
        if k<1 or k>self.n:
            self.logger.info(f"WARNING: k={k} is out of bound")
            
            # if we're in the training phase, we return a large negative reward and stop the episode
            if self.test_env is False:
                done = True
                n_evals = 0
                reward = -MAX_INT
                stop = True
            # if we're in the test phase, just clip k back to the range and continue
            else:
                k = np.clip(k,1,self.n)
                
        if stop is False:                                
            # flip k bits
            y, f_y, n_evals = self.x.mutate_rls(k, self.rng)         

            # update x
            if self.x.fitness <= y.fitness:
                self.x = y

            # update total number of evaluations        
            self.total_evals += n_evals

            # check stopping criteria        
            done = (self.total_evals>=self.instance.max_evals) or (self.x.is_optimal())        

            # calculate reward        
            imp = self.x.fitness - fitness_before_update
            if self.reward_choice=='imp_div_evals':        
                reward = imp / n_evals
            elif self.reward_choice=='imp_div_evals_new':            
                reward = (self.x.fitness - fitness_before_update - 0.5) / n_evals
            elif self.reward_choice=='imp_minus_evals':
                reward = self.x.fitness - fitness_before_update - n_evals
            elif self.reward_choice=='minus_evals':
                reward = -n_evals
            elif self.reward_choice=='minus_evals_normalised':
                reward = -n_evals / self.max_evals            
            elif self.reward_choice=='imp_minus_evals_normalised':
                reward = (self.x.fitness - fitness_before_update - n_evals) / self.max_evals
            elif self.reward_choice=='imp':
                reward = self.x.fitness - fitness_before_update
            self.log_reward.append(reward)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_k.append(k)        

        # update logs
        self.log_k.append(k)
        self.log_fx.append(self.x.fitness)
        self.log_reward.append(reward)
                    
        returned_info = {"msg": "", "values":{}}
        if done:            
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""    
            msg += "Episode done: n=%d; obj=%d; init_obj=%d; evals=%d; max_evals=%d; steps=%d; k_min=%.1f; k_max=%.1f; k_mean=%.1f; R=%.4f" % (self.n, self.x.fitness, self.init_obj, self.total_evals, self.max_evals, self.c_step, min(self.log_k), max(self.log_k), sum(self.log_k)/len(self.log_k), sum(self.log_reward))      
            #self.logger.info(msg) 
            returned_info['msg'] = msg
            returned_info['values'] = {'n':int(self.n), 
                                        'obj': int(self.x.fitness), 
                                        'init_obj': int(self.init_obj), 
                                        'evals': int(self.total_evals), 
                                        'max_evals': int(self.max_evals), 
                                        'steps': int(self.c_step), 
                                        'k_min': float(min(self.log_k)), 
                                        'k_max': float(max(self.log_k)), 
                                        'k_mean': float(sum(self.log_k)/len(self.log_k)), 
                                        'R': float(sum(self.log_reward)),
                                        'log_k': [int(x) for x in self.log_k],
                                        'log_fx':[int(x) for x in self.log_fx], 
                                        'log_reward': [float(x) for x in self.log_reward]}
        
        return self.get_state(), reward, done, returned_info

    def close(self) -> bool:
        """
        Close Env

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation
        """        
        return True

    
class RLSEnvDiscreteK(RLSEnv):
    """
    RLS environment where the choices of k is discretised
    """
    def __init__(self, config, test_env=False):
        super(RLSEnvDiscreteK, self).__init__(config, test_env)
        assert 'action_choices' in config, "Error: action_choices must be specified in benchmark's config"
        assert isinstance(self.action_space, gym.spaces.Discrete), "Error: action space must be discrete"
        assert self.action_space.n == len(config['action_choices']), "Error: action space's size (%d) must be equal to the len(action_choices) (%d)" % (self.action_space.n, len(config['action_choices']))        
        self.action_choices = config['action_choices']        
        
    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, list):
            assert len(action)==1
            action = action[0]
        return super(RLSEnvDiscreteK, self).step(self.action_choices[action])