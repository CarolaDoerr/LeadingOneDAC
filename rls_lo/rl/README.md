This folder contains scripts for running DAC-DDQN experiments on RLS for LeadingOne.

#### Code: 
Besides the DACBench code, all neccessary source code are in `scripts`.

#### Data: 
DDQN results used in the paper are available in `experiments/results`.


#### Installation:
Please follow the installation guidelines of DACBench (see `README_DACBench.md` in the repo's home folder)

#### Run experiments:

- Steps to run a DDQN training:

```
export DAC=<DACBench_repo_folder>
source $DAC/start.sh
cd $DAC/rls_lo/rl/experiments/
mkdir ddqn
python ../scripts/run_experiment.py --out-dir ddqn/ --setting-file train_conf.yml
```

The command above will train a DDQN agent with settings specified in `experiments/train_conf.yml` (current setting: `n=50, k=3`, `evenly_spread`). To switch to another setting or to change the hyper-parameters of DDQN, please change the options inside `experiments/train_conf.yml` accordingly. For example, to train a DDQN agent with `n=150`, `k=3`, and with `initial_segment` portfolio setting, you can update the following fields:

```
bench:
    action_choices:         [1,2,3]
    instance_set_path:      "lo_rls_150"    
```

- Steps to evaluate a trained DDQN policy:

```
python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy DQNPolicy --policy-params results/n50_evenly_spread/k3/trained_ddqn/best.pt --out results/n50_evenly_spread/k3/dqn.pkl >results/n50_evenly_spread/k3/ddqn.txt
```
The command above will do 2000 repeated runs of RLS on LeadingOne using the trained DDQN agent located in `n50_evenly_spread/k3/ddqn/best.pt`

- Steps to evaluate random/optimal policies:
```
python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy RandomPolicy --out results/n50_evenly_spread/k3/random.pkl >results/n50_evenly_spread/k3/random.txt

python ../scripts/run_policy.py --n-runs 2000 --bench test_conf.yml --policy RLSOptimalDiscretePolicy --out results/n50_evenly_spread/k3/optimal.pkl >results/n50_evenly_spread/k3/optimal.txt
```
