from tensorforce.agents import Agent
from .game import Game
from tqdm import trange
from multiprocessing import Process
import numpy as np
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
    nr_actions = dict(type='int', num_values=6)
    quant_actions = dict(type='int', num_values=8)
    cont_actions = dict(type='int', num_values=3)
    nr_agents = generate_agents(nplayers, nr_actions, environment, agent_path, agent)
    quant_agents = generate_agents(nplayers, quant_actions, environment, agent_path, agent)
    cont_agents = generate_agents(nplayers, cont_actions, environment, agent_path, agent, reward=True)
    run_nr = 0
    for _ in range(i):
        for j, agent in enumerate(nr_agents):
            agent.reset()
            quant_agents[j].reset()
            cont_agents[j].reset()
        states = environment.reset()
        rewards = {i: 0 for i in range(nplayers)}
        greedy_rewards = []
        greedy = run_nr % 5 == 0
        while not environment.terminal:
            # try:
            nr_agent = nr_agents[environment.current_player]
            quant_agent = quant_agents[environment.current_player]
            cont_agent = cont_agents[environment.current_player]
            nr_action = nr_agent.act(states=states, deterministic=greedy, independent=greedy)
            states = environment.execute_nr(nr_action)
            quant_action = quant_agent.act(states=states, deterministic=greedy, independent=greedy)
            states = environment.execute_quant(quant_action)
            cont_action = cont_agent.act(states=states, deterministic=greedy, independent=greedy)
            states, terminal, reward = environment.execute(cont_action=cont_action)
            if not greedy:
                nr_agent.observe(terminal=terminal, reward=reward)
                quant_agent.observe(terminal=terminal, reward=reward)
                cont_agent.observe(terminal=terminal, reward=reward)
            else:
                rewards[environment.current_player] += reward
            # except Exception:
            #     print(f"ENV {environment.state}")
            #     print(f"ACT {actions}")
            #     print(states)
            #     raise
        if greedy:
            greedy_rewards.append(sum(list(rewards.values())))
            greedy_rewards = greedy_rewards[-10:]
            print(run_nr, np.mean(greedy_rewards))
        run_nr += 1


def generate_agents(nplayers, actions, environment, agent_path, agent, reward=False):
    if reward:
        agents = [Agent.create(agent=f'{agent_path}/{agent}',
                               environment=environment,
                               actions=actions,
                               summarizer=dict(
                                   directory=f'summaries/{agent[:-5]}',
                                   summaries=['reward']),
                               saver=dict(
                                   directory=f'checkpoints/{agent[:-5]}',
                                   frequency=50)
                               ) for i in range(nplayers)]
    else:
        agents = [Agent.create(agent=f'{agent_path}/{agent}',
                               environment=environment,
                               actions=actions,
                               saver=dict(
                                   directory=f'checkpoints/{agent[:-5]}',
                                   frequency=50)
                               ) for i in range(nplayers)]
    return agents


if __name__ == "__main__":
    main(4)
