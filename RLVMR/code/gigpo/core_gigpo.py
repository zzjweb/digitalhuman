"""
Core functions to implement GiGPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO
"""

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order
    all_returns = np.zeros_like(rewards)
    for i, uid in enumerate(traj_uids):
        traj_indices = np.where(traj_uids == uid)[0]
        idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
        all_returns[i] = returns_by_traj[uid][idx_in_traj]
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm"
                                   ):
    
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode-level group reward
    episode_advantages = episode_norm_reward(token_level_rewards, eos_mask, index, epsilon, remove_std)
    
    # Compute step_group_uids
    step_group_uids = build_step_group(anchor_obs, index)

    # Compute step-level group reward
    step_advantages = step_norm_reward(step_rewards, eos_mask, step_group_uids, epsilon, remove_std)

    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        eos_mask: torch.Tensor,
                        index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of corresponding indices for each observation
    summarize : bool
        Whether to summarize the group sizes (default: True)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size = []
    # Process each unique index
    for idx in unique_indices:
        # Get all observations for this index using np.where
        indices = np.where(index == idx)[0]
        obs_group = anchor_obs[indices]
        
        # Create clusters for identical observations
        clusters = defaultdict(list)
        for i, obs in enumerate(obs_group):
            clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
        
        # Assign unique step_group_uid to each cluster
        for obs, original_indices in clusters.items():
            # Generate a UUID for this cluster
            uid = str(uuid.uuid4())
            
            # Assign the same step_group_uid to all elements in this cluster
            group_size.append(len(original_indices))
            for original_idx in original_indices:
                step_group_uids[original_idx] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      eos_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = eos_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
    
    return step_advantages

