import torch
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from PIL import Image
from verl import DataProto

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto) -> DataProto:
    size_divisor_ref = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
    size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    if bs % size_divisor != 0:
        remainder = bs % size_divisor
        
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    else:
        adjusted_batch = data

    return adjusted_batch


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, and success based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid

