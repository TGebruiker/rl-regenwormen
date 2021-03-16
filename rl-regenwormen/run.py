from tensorforce.agents import Agent
from .game import Game
from tqdm import trange
from multiprocessing import Process
import os


def main():
    i = 500
    nplayers = 1
    agent_path = "agents"
    for n,agent in enumerate(os.listdir(agent_path)):
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
                current_player = environment.current_player
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
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


if __name__ == "__main__":
    main(4)
