from tensorforce.agents import Agent
from .game import Game
from progress.bar import IncrementalBar as Bar


def main(nplayers):
    print("creating environment...")
    environment = Game(nplayers)
    print("creating agents...")
    agents = [Agent.create(agent='ppo',
                           batch_size=10,
                           learning_rate=1e-3,
                           environment=environment
                           #    summarizer=dict(
                           #        directory='../summaries',
                           #        # list of labels, or 'all'
                           #        labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
                           ) for i in range(nplayers)]

    print("starting training...")
    i = 100
    bar = Bar('Training', max=i)
    for _ in range(100):
        states = environment.reset()
        terminal = False
        while not terminal:
            agent = agents[environment.current_player]
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
        bar.next()
    bar.finish()


if __name__ == "__main__":
    main(4)
