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

```bash
wandb login
python training.py
```

Provide your wandb API key when prompted. (Get one from https://wandb.com)

### Results

Our neural networks:
- Policy Network(actor) for Predators:
PolicyNetwork(
  (fc1): Linear(in_features=16, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=5, bias=True)
)

- Policy Network(actor) for Prey:
PolicyNetwork(
  (fc1): Linear(in_features=14, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=5, bias=True)
)

- Q Network(critic) for MADDPG:
MADDPGQNetwork(
  (fc1): Linear(in_features=63, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)

- Q Network(critic) for DDPG:
DDPGQNetwork(
  (fc1): Linear(in_features=19, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)

### References

- [PettingZoo-MPE](https://github.com/Farama-Foundation/PettingZoo)
- [MADDPG](http://arxiv.org/abs/1706.02275)
- [DDPG](http://arxiv.org/abs/1509.02971)