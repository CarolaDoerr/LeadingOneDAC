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