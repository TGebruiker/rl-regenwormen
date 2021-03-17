from tensorforce.agents import Agent
from .game import Game
from tqdm import trange
from multiprocessing import Process
import os


def test():
    environment = Game(1, show=True)
    agents = [Agent.load(directory='checkpoints/ddqn', format='checkpoint', environment=environment)]
    states = environment.reset()
    while True:
        agent = agents[environment.current_player]
        actions = agent.act(states=states, independent=True, deterministic=True)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)


def main():
    single = True
    i = 500000
    nplayers = 1
    agent_path = "agents"
    if single:
        run(0, agent_path, "ddqn.json", 1, i)
    else:
        for n, agent in enumerate(os.listdir(agent_path)):
            p = Process(target=run, args=(n, agent_path, agent, nplayers, i))
            p.start()


def run(n, agent_path, agent, nplayers, i):
    environment = Game(nplayers)
    agents = [Agent.create(agent=f'{agent_path}/{agent}',
                           environment=environment,
                           summarizer=dict(
                               directory=f'summaries/{agent[:-5]}',
                               summaries=['reward']),
                           saver=dict(
                               directory=f'checkpoints/{agent[:-5]}',
                               frequency=50)
                           ) for i in range(nplayers)]

    for _ in trange(i, desc=agent[:-5], position=n):
        for agent in agents:
            agent.reset()
        states = environment.reset()
        while not environment.terminal:
            try:
                agent = agents[environment.current_player]
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
            except Exception:
                print(f"ENV {environment.state}")
                print(f"ACT {actions}")
                print(states)
                raise


if __name__ == "__main__":
    main(4)
