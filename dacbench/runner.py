from dacbench import benchmarks
from dacbench.wrappers import PerformanceTrackingWrapper
from dacbench.logger import Logger
import seaborn as sb
from pathlib import Path

sb.set_style("darkgrid")
current_palette = list(sb.color_palette())


def run_benchmark(env, agent, num_episodes, logger=None):
    """
    Run single benchmark env for a given number of episodes with a given agent

    Parameters
    -------
    env : gym.Env
        Benchmark environment
    agent
        Any agent implementing the methods act, train and end_episode (see AbstractDACBenchAgent below)
    num_episodes : int
        Number of episodes to run
    logger : dacbench.logger.Logger: logger to use for logging. Not closed automatically like env
    """
    if logger is not None:
        logger.reset_episode()
        logger.set_env(env)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        reward = 0
        while not done:
            action = agent.act(state, reward)
            next_state, reward, done, _ = env.step(action)
            agent.train(next_state, reward)
            state = next_state
            if logger is not None:
                logger.next_step()
        agent.end_episode(state, reward)

        if logger is not None:
            logger.next_episode()
    env.close()


def run_dacbench(results_path, agent_method, num_episodes, bench=None, seeds=None):
    """
    Run all benchmarks for 10 seeds for a given number of episodes with a given agent and save result

    Parameters
    -------
    bench
    results_path : str
        Path to where results should be saved
    agent_method : function
        Method that takes an env as input and returns an agent
    num_episodes : int
        Number of episodes to run for each benchmark
    seeds : list[int]
        List of seeds to runs all benchmarks for. If None (default) seeds [1, ..., 10] are used.
    """

    if bench is None:
        bench = map(benchmarks.__dict__.get, benchmarks.__all__)
    else:
        bench = [getattr(benchmarks, b) for b in bench]

    seeds = seeds if seeds is not None else range(10)
    for b in bench:
        print(f"Evaluating {b.__name__}")
        for i in seeds:
            print(f"Seed {i}/10")
            bench = b()
            try:
                env = bench.get_benchmark(seed=i)
            except:
                continue

            logger = Logger(
                experiment_name=f"seed_{i}",
                output_path=Path(results_path) / f"{b.__name__}",
            )
            perf_logger = logger.add_module(PerformanceTrackingWrapper)
            logger.add_benchmark(bench)
            logger.set_env(env)

            env = PerformanceTrackingWrapper(env, logger=perf_logger)
            agent = agent_method(env)
            logger.add_agent(agent)

            run_benchmark(env, agent, num_episodes, logger)

            logger.close()
