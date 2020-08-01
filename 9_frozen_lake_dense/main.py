# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import gym
import gym.spaces
from collections import namedtuple
from loguru import logger

# %%
env = gym.make("FrozenLake-v0", is_slippery=False)

# %%
class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OneHotWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (env.observation_space.n,), dtype=np.float32
        )

    def observation(self, observation):
        r = np.copy(self.observation_space.low)
        r[observation] = 1.0
        return r


env = OneHotWrapper(env)

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
num_hiddens = 128

net = nn.Sequential(
    nn.Linear(obs_size, num_hiddens), nn.ReLU(), nn.Linear(num_hiddens, n_actions)
)

objective = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# %%
def select_action(state):
    state = torch.tensor([state])
    act_probs = F.softmax(net(state), dim=1)
    act_probs = act_probs.data.numpy()[0]
    action = np.random.choice(len(act_probs), p=act_probs)
    return action


reward_mean = 0
reward_goal = 0.8

Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])

episode_steps = []
episode_rewards = 0.0

gamma = 0.9

batch = []
batch_size = 100

state = env.reset()

while reward_mean < reward_goal:
    action = select_action(state)
    next_state, reward, done, _ = env.step(action)

    episode_steps.append(EpisodeStep(observation=state, action=action))
    episode_rewards += reward

    if done:
        batch.append(Episode(reward=reward, steps=episode_steps))
        next_state = env.reset()

        episode_steps = []
        episode_rewards = 0.0

        if len(batch) == batch_size:
            reward_mean = float(np.mean([s.reward for s in batch]))
            candidates = elite_batch + batch
            returnG = [s.reward * (gamma ** len(s.steps)) for s in candidates]

            reward_bound = np.percentile(returnG, 30)

            train_obs = []
            train_act = []
            elite_batch = []

            for example, reward in zip(candidates, returnG):
                if reward > reward_bound:
                    train_obs.extend([step.observation for step in example.steps])
                    train_act.extend([step.action for step in example.steps])
                    elite_batch.append(example)

            if len(elite_batch) != 0:
                state_t = torch.tensor(train_obs)
                acts_t = torch.LongTensor(train_act)
                optimizer.zero_grad()
                scores = net(state_t)
                loss = objective(scores, acts_t)
                loss.backward()
                optimizer.step()

                logger.info(f"Loss={loss.item()}, mean_reward={reward_mean}")

            batch = []

    state = next_state


# %%
test_env = OneHotWrapper(gym.make("FrozenLake-v0", is_slippery=False))
state = test_env.reset()
test_env.render()

is_done = False

while not is_done:
    action = select_action(state)
    state, reward, is_done, _ = test_env.step(action)
    test_env.render()

logger.info(f"Final reward: {reward}")
