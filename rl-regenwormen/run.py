from tensorforce.agents import Agent
from .game import Game
from progress.bar import IncrementalBar as Bar
from tensorforce.execution import Runner
import numpy as np

import json


def main(nplayers):
    print("creating environment...")
    environment = Game(nplayers)
    print("creating agents...")
    run_no_runner(environment, nplayers)


def run_runner(environment):
    agent = Agent.create(agent='dqn',
                         memory=10,
                         batch_size=4,
                         environment=environment,
                         summarizer=dict(
                             directory='summaries',
                             labels='all')
                         )
    runner = Runner(
        agent=agent,
        environment=environment,
        # num_parallel=5, remote='multiprocessing'
    )

    runner.run(num_episodes=30000)

    runner.run(num_episodes=10000, evaluation=True)


def run_no_runner(environment, nplayers):
    #with open("rl-regenwormen/agent.json", 'r') as fp:
    #    agent = json.load(fp=fp)

    agents = [Agent.create(agent='ppo',
                           batch_size=100,
                           learning_rate=1e-3,
                           exploration=0.2,
                           environment=environment,
                           summarizer=dict(
                               directory='summaries',
                               summaries='all')
                           ) for i in range(nplayers)]

    print("starting training...")
    i = 10000000
    bar = Bar('Training', max=i)
    rewards = {i: 0 for i in range(nplayers)}
    rewards_total = {i: [] for i in range(nplayers)}
    for episode in range(30000):
        for agent in agents:
            agent.reset()
        states = environment.reset()
        terminal = False
        while not terminal:
            try:
                agent = agents[environment.current_player]
                current_player = environment.current_player
                actions = agent.act(states=states)
                #print(actions)
                states, terminal, reward = environment.execute(actions=actions)
                rewards[environment.current_player] += reward
                rewards_total[environment.current_player] += [reward]
                rewards_total[environment.current_player] = rewards_total[environment.current_player][-300:]
                end_of_roll = environment.current_player != current_player
                agent.observe(terminal=end_of_roll, reward=reward)
                if terminal:
                    for agent2 in agents:
                        if agent2 != agent:
                            actions = agent2.act(states=states)
                            states, terminal, reward = environment.execute(actions=actions)
                            agent2.observe(terminal=True, reward=reward)
            except:
                print(f"ENV {environment.state}")
                print(f"ACT {actions}")
                print(states)
                raise
        names = ["lola", "henry de muis", "pykel", "flo"]
        print({names[k]: (int(v * 100)/100,  int(np.mean(rewards_total[k]) * 100) / 100) for k, v in rewards.items()})
        rewards = {i: 0 for i in range(nplayers)}
        bar.next()
    bar.finish()


if __name__ == "__main__":
    main(4)
