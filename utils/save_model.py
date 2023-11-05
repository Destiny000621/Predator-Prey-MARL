import torch
import os

def save_maddpg(maddpg_agent, directory):
    """
    Saves the MADDPG agent's actor and critic networks.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for idx, policy_net in enumerate(maddpg_agent.predator_policy_nets):
        torch.save(policy_net.state_dict(), os.path.join(directory, f'predator_policy_net_{idx}.pth'))
        
    torch.save(maddpg_agent.predator_q_net.state_dict(), os.path.join(directory, 'predator_q_net.pth'))
    
    for idx, target_policy_net in enumerate(maddpg_agent.target_predator_policy_nets):
        torch.save(target_policy_net.state_dict(), os.path.join(directory, f'target_predator_policy_net_{idx}.pth'))
    
    torch.save(maddpg_agent.target_predator_q_net.state_dict(), os.path.join(directory, 'target_predator_q_net.pth'))

def save_ddpg(ddpg_agent, agent_name, directory):
    """
    Saves the DDPG agent's actor and critic networks.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    policy_filename = f"{agent_name}_policy_net.pth"
    ddpg_q_filename = f"{agent_name}_ddpg_q_net.pth"
    target_policy_filename = f"{agent_name}_target_policy_net.pth"
    target_ddpg_q_filename = f"{agent_name}_target_ddpg_q_net.pth"

    policy_path = os.path.join(directory, policy_filename)
    ddpg_q_path = os.path.join(directory, ddpg_q_filename)
    target_policy_path = os.path.join(directory, target_policy_filename)
    target_ddpg_q_path = os.path.join(directory, target_ddpg_q_filename)
    
    torch.save(ddpg_agent.policy_net.state_dict(), policy_path)
    torch.save(ddpg_agent.ddpg_q_net.state_dict(), ddpg_q_path)
    torch.save(ddpg_agent.target_policy_net.state_dict(), target_policy_path)
    torch.save(ddpg_agent.target_ddpg_q_net.state_dict(), target_ddpg_q_path)

def save_dqn(dqn_agent, agent_name, directory):
    """
    Saves the DQN agent's network.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    local_filename = f"{agent_name}_qnetwork_local.pth"
    target_filename = f"{agent_name}_qnetwork_target.pth"

    local_path = os.path.join(directory, local_filename)
    target_path = os.path.join(directory, target_filename)
    
    torch.save(dqn_agent.qnetwork_local.state_dict(), local_path)
    torch.save(dqn_agent.qnetwork_target.state_dict(), target_path)


def save_iac(iac_agent, agent_name, directory):
    """
    Saves the IAC agent's network.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    actor_filename = f"{agent_name}_actor_net.pth"
    critic_filename = f"{agent_name}_critic_net.pth"

    actor_path = os.path.join(directory, actor_filename)
    critic_path = os.path.join(directory, critic_filename)

    torch.save(iac_agent.actor_net.state_dict(), actor_path)
    torch.save(iac_agent.critic_net.state_dict(), critic_path)

