# PYTHON_ARGCOMPLETE_OK
import argparse
import re
from collections.abc import Generator
from dataclasses import dataclass

import argcomplete
import gymnasium as gym
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
ENV_NAME = "CartPole-v1"

logger = gym.logger
logger.set_level(gym.logger.INFO)


class Policy(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(obs_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.model(x)


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: list[EpisodeStep]


@torch.no_grad()
def iterate_batches(
    fabric: L.Fabric, env: gym.Env, policy: Policy, batch_size: int = BATCH_SIZE
) -> Generator[list[Episode], None, None]:
    batch: list[Episode] = []
    episode_reward = 0.0
    episode_steps: list[EpisodeStep] = []
    obs, _ = env.reset()
    softmax = nn.Softmax(dim=1)
    policy.eval()

    while True:
        obs_vec = torch.tensor(obs, dtype=torch.float32)
        obs_vec = fabric.to_device(obs_vec)
        act_probs_vec = softmax(policy(obs_vec.unsqueeze(0)))
        act_probs = act_probs_vec.cpu().numpy()[0]
        action = int(np.random.choice(len(act_probs), p=act_probs))
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done or is_trunc:
            episode = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(episode)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch.clear()
                policy.eval()
        obs = next_obs


def filter_batch(
    fabric: L.Fabric, batch: list[Episode], percentile: float = PERCENTILE
) -> tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = [episode.reward for episode in batch]
    reward_bound = float(np.percentile(rewards, percentile))
    rewards_mean = float(np.mean(rewards))

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        obs_list.extend([step.observation for step in episode.steps])
        act_list.extend([step.action for step in episode.steps])
    obs_vec = fabric.to_device(torch.FloatTensor(np.stack(obs_list)))
    act_vec = fabric.to_device(torch.LongTensor(act_list))
    return obs_vec, act_vec, rewards_mean, reward_bound


def pos_int(value: str) -> int:
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Invalid value {value}: must be a positive number"
        )
    return int_value


def percentile_type(value: str) -> float:
    float_value = float(value)
    if 0 < float_value <= 100:
        return float_value
    raise argparse.ArgumentTypeError("Value must be a float between 0 and 100")


def device_type(value: str):
    if re.match(r"-?\d+", value):
        int_value = int(value)
        if int_value == 0 or int_value < -1:
            raise argparse.ArgumentTypeError(
                "Must be an integer greater than 0 or equal to -1"
            )
        return int_value
    elif value == "auto":
        return value
    elif re.match(r"\[(\d+,\s)*\d\]", value):
        return eval(value)
    else:
        raise argparse.ArgumentTypeError("Invalid argument")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Cross-entropy method on Cartpole-v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hidden-size", type=pos_int, default=HIDDEN_SIZE)
    parser.add_argument("--batch-size", type=pos_int, default=BATCH_SIZE)
    parser.add_argument("--percentile", type=percentile_type, default=PERCENTILE)
    parser.add_argument("--env-name", default=ENV_NAME)

    parser.add_argument(
        "--precision", choices=["16-mixed", "32-true"], default="16-mixed", help="-"
    )
    parser.add_argument(
        "--strategy", choices=("ddp", "dp", "auto"), default="auto", help="-"
    )
    parser.add_argument("--num-nodes", type=pos_int, default=1, help="-")
    parser.add_argument("--devices", type=device_type, default="auto", help="-")
    parser.add_argument(
        "--accelerator", choices=("gpu", "cpu", "auto"), default="auto", help="-"
    )
    argcomplete.autocomplete(parser)
    args, _ = parser.parse_known_args(args)

    fabric = L.Fabric(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        precision=args.precision,
    )
    fabric.launch()

    env = gym.make(args.env_name)
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = int(env.action_space.n)

    policy = Policy(obs_size, args.hidden_size, num_actions)
    # logger.info(repr(policy))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01, amsgrad=True)

    policy, optimizer = fabric.setup(policy, optimizer)

    writer = torch.utils.tensorboard.SummaryWriter(
        comment=f"-{args.env_name.split('-')[0].lower()}"
    )

    for iter_num, batch in enumerate(iterate_batches(env, policy, args.batch_size)):
        obs_vec, act_vec, rewards_mean, reward_bound = filter_batch(
            batch, args.percentile
        )
        policy.train()
        optimizer.zero_grad()
        action_logits_vec = policy(obs_vec)
        with fabric.autocast():
            loss = F.cross_entropy(action_logits_vec, act_vec)
        fabric.backward(loss)
        optimizer.step()
        logger.info(
            f"{iter_num}: {loss=:.3f}, {rewards_mean=:.1f}, {reward_bound=:.1f}"
        )
        writer.add_scaler("loss", loss.item(), iter_num)
        writer.add_scalar("reward_mean", rewards_mean, iter_num)
        writer.add_scalar("reward_bound", reward_bound, iter_num)
        if rewards_mean >= env.spec.reward_threshold:
            logger.info("Solved!")
            break
    writer.close()


if __name__ == "__main__":
    main()
