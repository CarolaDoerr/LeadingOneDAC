class RLSEnv(AbstractEnv):
    """
    Environment for RLS with step size
    for both OneMax and LeadingOne problems
    """

    def __init__(self, config) -> None:
        """
        Initialize RLSEnv

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(RLSEnv, self).__init__(config)        
        self.logger = logging.getLogger(self.__str__())     

        self.name = config.name   
        
        # name of reward function
        assert config.reward_choice in ['imp_div_evals', 'imp_div_evals_new', 'imp_minus_evals', 'minus_evals', 'imp', 'minus_evals_normalised', 'imp_minus_evals_normalised']
        self.reward_choice = config.reward_choice        
        #print("Reward choice: " + self.reward_choice)        

        # parameters of RLS
        self.problem = globals()[config.problem]                

        # names of all variables in a state
        self.state_description = config.observation_description
        self.state_var_names = [s.strip() for s in config.observation_description.split(',')]

        # functions to get values of the current state from histories 
        # (see reset() function for those history variables)        
        self.state_functions = []
        for var_name in self.state_var_names:
            if var_name == 'n':
                self.state_functions.append(lambda: self.n)
            elif var_name in ['k','k_base2_interval_start']:
                self.state_functions.append(lambda his='history_'+var_name: vars(self)[his][-1])
            elif "_{t-" in var_name:
                k = int(var_name.split("_{t-")[1][:-1]) # get the number in _{t-<number>}
                name = var_name.split("_{t-")[0] # get the variable name (lbd, lbd1, etc)
                self.state_functions.append(lambda his='history_'+name: vars(self)[his][-k])
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta_f(x)":
                self.state_functions.append(lambda: self.history_fx[-1] - self.history_fx[-2])
            elif var_name == "optimal_k":
                self.state_functions.append(lambda: int(self.n/(self.history_fx[-1]+1)))
            elif var_name == 'optimal_k_base2_interval_start':
                self.state_functions.append(lambda: int(math.log(int(self.n/(self.history_fx[-1]+1)), 2)))
            else:
                raise Exception("Error: invalid state variable name: " + var_name)
        
        # names of all variables in an action        
        self.action_description = config.action_description
        self.action_var_names = [s.strip() for s in config.action_description.split(',')] # names of 
        for name in self.action_var_names:
            assert name in ['k', 'k_base2_interval_start'], "Error: invalid action variable name: " + name
        
        # the random generator used by RLS
        if 'seed' in config:
            seed = config.seed
        else:
            seed = None
        self.rng = np.random.default_rng(seed)   

    def seed(self, seed=None):
        super(OneLLEnv, self).seed(seed)
        self.rng = np.random.default_rng(seed)
    
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

        # reset histories (not all of those are used at the moment)        
        self.history_k = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) # either this one or the next two (history_lbd1, history_lbd2) are used, depending on our configuration
        self.history_k_base2_interval_start = deque([-1]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_fx = deque([self.x.fitness]*HISTORY_LENGTH, maxlen=HISTORY_LENGTH) 

        # for debug only
        self.ks = []
        self.rewards = []     
        self.fs = []
        self.init_obj = self.x.fitness 
        
        return self.get_state()

    def get_state(self):
        return np.asarray([f() for f in self.state_functions])
    
    def get_rls_params(self, action):
        """
        Get RLS parameters (k only for now) from an action

        Returns: k
            k: int
                number of bits being flipped during mutation
        """
        if (not isinstance(action, np.ndarray)) and (not isinstance(action, list)):
            if self.action_choices: # TODO: only support 1-d discrete action space 
                action = self.action_choices[self.action_var_names[0]][action]
            action = [action]
        i = 0
        for var_name in self.action_var_names:
            if var_name == 'k':
                k = np.clip(action[i], 1, self.n)
            elif var_name == 'k_base2_interval_start': 
                k = np.clip(rng.choice(range(2**action[i], 2**action[i]+1), size=1), 1, self.n)
            else: 
                raise Exception("Error: invalid action name" + var_name)
            i+=1

        return k

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

        k = self.get_rls_params(action) 
                
        # for logging
        fitness_before_update = self.x.fitness
        
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
        self.rewards.append(reward)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_k.append(k)
        self.history_k_base2_interval_start.append(action)

        self.ks.append(k)
        self.fs.append(self.x.fitness)
                    
        returned_info = {"msg": "", "values":{}}
        if done:
            self.n_eps += 1
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""    
            msg += "Episode done: ep=%d; n=%d; obj=%d; init_obj=%d; evals=%d; max_evals=%d; steps=%d; k_min=%.1f; k_max=%.1f; k_mean=%.1f; R=%.4f" % (self.n_eps, self.n, self.x.fitness, self.init_obj, self.total_evals, self.max_evals, self.c_step, min(self.ks), max(self.ks), sum(self.ks)/len(self.ks), sum(self.rewards))      
            #self.logger.info(msg) 
            returned_info['msg'] = msg
            returned_info['values'] = {'n':int(self.n), 
                                        'obj': int(self.x.fitness), 
                                        'init_obj': int(self.init_obj), 
                                        'evals': int(self.total_evals), 
                                        'max_evals': int(self.max_evals), 
                                        'steps': int(self.c_step), 
                                        'k_min': float(min(self.ks)), 
                                        'k_max': float(max(self.ks)), 
                                        'k_mean': float(sum(self.ks)/len(self.ks)), 
                                        'R': float(sum(self.rewards)),
                                        'fs': [int(x) for x in self.fs],
                                        'ks':[int(x) for x in self.ks], 
                                        'rewards': [float(x) for x in self.rewards]}
        
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