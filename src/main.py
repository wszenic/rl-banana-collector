import numpy as np

from agent import Agent
from collections import namedtuple
from unityagents import UnityEnvironment
import neptune.new as neptune

from config import *
import os


def main():
    env = UnityEnvironment(
        file_name="/Banana.app",
        no_graphics=True
    )

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    run = neptune.init(
        project="wsz/RL-bananas",
        api_token=os.getenv('NEPTUNE_TOKEN')
    )

    run['parameters'] = {
        'MID_1_SIZE': MID_1_SIZE,
        'MID_2_SIZE': MID_2_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'EPS_MIN': EPS_MIN,
        'EPS_MAX': EPS_MAX,
        'GAMMA': GAMMA,
        'TAU': TAU,
        'BATCH_SIZE': BATCH_SIZE
    }

    state = env_info.vector_observations[0]

    agent = Agent(state_size=len(state), action_size=brain.vector_action_space_size)
    env_feedback = namedtuple('env_feedback', ('state', 'action', 'reward', 'next_state', 'done'))

    scores = []

    epsilon_space = np.concatenate([
        np.linspace(EPS_MAX, EPS_MIN, EXPLORATORY_EPOCHS),
        np.repeat(EPS_MIN, (MAX_EPOCH - EXPLORATORY_EPOCHS))
        ])

    for episode in range(MAX_EPOCH):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        eps = epsilon_space[episode]
        while True:
            action = agent.act(state, eps)                       # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            env_response = env_feedback(state, action, env_info.rewards[0], env_info.vector_observations[0], env_info.local_done[0])

            agent.step(env_response)

            score += env_response.reward                                # update the score
            state = env_response.next_state                             # roll over the state to next time step
            if env_response.done:                                       # exit loop if episode finished
                run['score'].log(score)
                break
        scores.append(score)
        print(f"Episode {episode} | Score = {score} | Max score = {np.max(scores)} | Avg = {np.mean(scores[-10:]):.2f}")
    run.stop()
    env.close()


if __name__ == "__main__":
    main()

