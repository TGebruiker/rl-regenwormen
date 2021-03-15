from tensorforce.agents import Agent
from .game import Game
from progress.bar import IncrementalBar as Bar
from tensorforce.execution import Runner

import json


def main(nplayers):
    print("creating environment...")
    environment = Game(nplayers)
    print("creating agents...")
    run_no_runner(environment, 4)


def run_runner(env):
    with open("rl-regenwormen/agent.json", 'r') as fp:
        agent = json.load(fp=fp)
    runner = Runner(
        agent=agent,
        environment=env,
        # num_parallel=5, remote='multiprocessing'
    )

    runner.run(num_episodes=30000)

    runner.run(num_episodes=10000, evaluation=True)


def run_no_runner(environment, nplayers):
    with open("rl-regenwormen/agent.json", 'r') as fp:
        agent = json.load(fp=fp)

    agents = [Agent.create(agent='dqn',
                           memory=10,
                           batch_size=4,
                           environment=environment,
                           summarizer=dict(
                               directory='summaries',
                               labels='all')
                           ) for i in range(nplayers)]

    print("starting training...")
    i = 10000000
    bar = Bar('Training', max=i)
    rewards = {i: 0 for i in range(nplayers)}
    for _ in range(10000000):
        for agent in agents:
            agent.reset()
        states = environment.reset()
        terminal = False
        while not terminal:
            try:
                agent = agents[environment.current_player]
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                rewards[environment.current_player] += reward
                agent.observe(terminal=terminal, reward=reward)
            except EnvironmentError:
                print(f"ENV {environment.state}")
                print(f"ACT {actions}")
                print(states)
                raise
        print(rewards)
        bar.next()
    bar.finish()


if __name__ == "__main__":
    main(4)
