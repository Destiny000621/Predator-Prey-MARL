import torch
import os

def load_model(agent, filename):
    agent.qnetwork_local.load_state_dict(torch.load(filename))
    agent.qnetwork_local.eval() #TODO: check if this is necessary

def load_dqn(dqn_agent, agent_name, directory):
    """
    Loads the DQN agent's network from saved state dictionaries.
    """
    local_filename = f"{agent_name}_qnetwork_local.pth"
    target_filename = f"{agent_name}_qnetwork_target.pth"
    local_path = os.path.join(directory, local_filename)
    target_path = os.path.join(directory, target_filename)
    
    if os.path.isfile(local_path) and os.path.isfile(target_path):
        dqn_agent.qnetwork_local.load_state_dict(torch.load(local_path))
        dqn_agent.qnetwork_target.load_state_dict(torch.load(target_path))
        print(f"Local model loaded: {local_path}")
        print(f"Target model loaded: {target_path}")

    else:
        print("Model files not found.")


def load_iac(iac_agent, agent_name, directory):
    """
    Loads the IAC agent's network from saved state dictionaries.
    """
    actor_filename = f"{agent_name}_actor_network.pth"
    critic_filename = f"{agent_name}_critic_network.pth"
    actor_path = os.path.join(directory, actor_filename)
    critic_path = os.path.join(directory, critic_filename)

    if os.path.isfile(actor_path) and os.path.isfile(critic_path):
        iac_agent.actor_network.load_state_dict(torch.load(actor_path))
        iac_agent.critic_network.load_state_dict(torch.load(critic_path))
        print(f"Actor model loaded: {actor_path}")
        print(f"Critic model loaded: {critic_path}")

    else:
        print("Model files not found.")

def load_ddpg(ddpg_agent, agent_name, directory):
    """
    Loads the DDPG agent's actor (policy) and critic networks from saved state dictionaries.
    """
    policy_filename = f"{agent_name}_policy_net.pth"
    ddpg_q_filename = f"{agent_name}_ddpg_q_net.pth"
    target_policy_filename = f"{agent_name}_target_policy_net.pth"
    target_ddpg_q_filename = f"{agent_name}_target_ddpg_q_net.pth"

    policy_path = os.path.join(directory, policy_filename)
    ddpg_q_path = os.path.join(directory, ddpg_q_filename)
    target_policy_path = os.path.join(directory, target_policy_filename)
    target_ddpg_q_path = os.path.join(directory, target_ddpg_q_filename)

    if os.path.isfile(policy_path) and os.path.isfile(ddpg_q_path) and os.path.isfile(target_policy_path) and os.path.isfile(target_ddpg_q_path):
        ddpg_agent.policy_net.load_state_dict(torch.load(policy_path))
        ddpg_agent.ddpg_q_net.load_state_dict(torch.load(ddpg_q_path))
        ddpg_agent.target_policy_net.load_state_dict(torch.load(target_policy_path))
        ddpg_agent.target_ddpg_q_net.load_state_dict(torch.load(target_ddpg_q_path))
        print(f"Local model loaded: {policy_path}")
        print(f"Target model loaded: {ddpg_q_path}")
        print(f"Local model loaded: {target_policy_path}")
        print(f"Target model loaded: {target_ddpg_q_path}")

    else:
        print("Model files not found.")

def load_maddpg(maddpg_agent, directory):
    """
    Loads the MADDPG agent's actor and critic networks from saved state dictionaries.
    """
    for idx, policy_net in enumerate(maddpg_agent.predator_policy_nets):
        policy_net.load_state_dict(torch.load(os.path.join(directory, f'predator_policy_net_{idx}.pth')))
        
    maddpg_agent.predator_q_net.load_state_dict(torch.load(os.path.join(directory, 'predator_q_net.pth')))
    
    for idx, target_policy_net in enumerate(maddpg_agent.target_predator_policy_nets):
        target_policy_net.load_state_dict(torch.load(os.path.join(directory, f'target_predator_policy_net_{idx}.pth')))
    
    maddpg_agent.target_predator_q_net.load_state_dict(torch.load(os.path.join(directory, 'target_predator_q_net.pth')))