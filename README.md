# Predator-Prey-MARL
PyTorch Implementation of MADDPG and DDPG in a Multi-Agents Predator-Prey(MAPP) environment.

### Install Python Environment

Install Python environment with conda:

```bash
conda create -n pp_env python=3.10 -y
conda activate pp_env
pip install -r requirements.txt
```

### How to see the neural network architecture

Execute the following command:

```bash
cd ${HOME}/Predator-Prey-MARL/utils
python network.py
```

### How to Train the Agents

Execute the following command to train the agents:

1. MADDPG for Predators and DDPG for Prey

```bash
wandb login
python training_MADDPG.py
```

2. DDPG for Predators and Prey

```bash
wandb login
python training_DDPG.py
```

Provide your wandb API key when prompted. (Get one from https://wandb.com)

### Results


### References

- [PettingZoo-MPE](https://github.com/Farama-Foundation/PettingZoo)
- [MADDPG](http://arxiv.org/abs/1706.02275)
- [DDPG](http://arxiv.org/abs/1509.02971)