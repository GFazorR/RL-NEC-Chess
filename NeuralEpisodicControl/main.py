from torch.optim import Adam

from NEC import NeuralEpisodicControl
import numpy as np
import matplotlib.pyplot as plt
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from Chess import Chess
from scipy.stats import sem

colors = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'purple', 'pink']
model_names = ['Deep Q Learning', 'Deep Sarsa', 'Deep Expected Sarsa']


def test_models(env_name, env, n_steps, n_experiments=5,
                n_episodes=100, lr=1e-3):
    fig, ax = plt.subplots(1, len(n_steps), sharex='col', sharey='row', figsize=(16, 16))

    fig.suptitle(f'Learning curves for {env_name}')
    for i, n in enumerate(n_steps):
        experiments_rewards = np.zeros(shape=(n_experiments, n_episodes))
        print(f'Start experiments with n_steps: {n}')
        for j in range(n_experiments):
            # Initialize Network

            q_network = QNetwork()
            replay_buffer = ReplayBuffer(10000)
            optimizer = Adam

            agent = NeuralEpisodicControl(
                env,
                q_network,
                replay_buffer,
                optimizer,
                n_steps=n
            )

            rewards = agent.train(n_episodes)
            experiments_rewards[j] = rewards
            print(f'\tEnd of experiment {j} with parameter n_steps: {n}')

        mean_cumulative_costs = experiments_rewards.mean(axis=0)
        sem_error = sem(experiments_rewards, axis=0)

        mean_plus_error = np.add(mean_cumulative_costs, sem_error)
        mean_minus_error = np.subtract(mean_cumulative_costs, sem_error)
        ax[i].plot(np.arange(0, n_episodes, 1),
                   mean_cumulative_costs,
                   color=colors[i])
        ax[i].fill_between(np.arange(0, n_episodes, 1),
                           mean_minus_error,
                           mean_plus_error,
                           color=colors[i],
                           alpha=.2)
        ax[i].title.set_text(f'{model_names[k]} with steps: {n}')

        ax[i].set_xlabel('Episodes')
        ax[i].set_ylabel('Average Cumulative Cost')

    fig.savefig(f'learning-curve-{env_name}.png')
    fig.show()


# TODO
if __name__ == '__main__':
    test_models(
        env_name='Chess',
        env=Chess(),
        n_steps=[1, 3, 5],
        n_episodes=50,
        n_experiments=3
    )