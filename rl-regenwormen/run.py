from tensorforce.agents import Agent
from .game import Game
from tqdm import trange
from multiprocessing import Process
import os


def main():
    single = True
    i = 500000
    nplayers = 1
    agent_path = "agents"
    if single:
        run(0, agent_path, "ppo.json", 1, i)
    for n, agent in enumerate(os.listdir(agent_path)):
        p = Process(target=run, args=(n, agent_path, agent, nplayers, i))
        p.start()


def run(n, agent_path, agent, nplayers, i):
    environment = Game(nplayers)
    agents = [Agent.create(agent=f'{agent_path}/{agent}',
                           environment=environment,
                           summarizer=dict(
                               directory='summaries',
                               summaries='all'),
                           saver=dict(
                               directory=f'checkpoints/{agent[:-5]}',
                               frequency=100)
                           ) for i in range(nplayers)]

    for i in trange(i, desc=agent[:-5], position=n):
        for agent in agents:
            agent.reset()
        states = environment.reset()
        terminal = False
        while not terminal:
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
