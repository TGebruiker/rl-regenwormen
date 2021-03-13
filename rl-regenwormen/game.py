from tensorforce.environments import Environment
import numpy as np
from math import floor
from random import randint


class Game(Environment):
    def __init__(self, nplayers=4):
        super().__init__()
        self.nplayers = nplayers
        self.state = self.reset()

    def reset(self):
        self.current_player = 0
        self.state = {'stone_pos': [0] * 16,
                 'stone_lock': [0] * 16,
                 'dice_value': [0] * 8,
                 'dice_lock': [0] * 8}
        return self.state

    def states(self):
        # 15 stones and 8 dice
        return dict(stone_pos=dict(type='int', shape=(16,), num_values=self.nplayers+1),
                    stone_lock=dict(type='int', shape=(16,), num_values=2),
                    dice_value=dict(type='int', shape=(8,), num_values=6),
                    dice_lock=dict(type='int', shape=(8,), num_values=2))

    def actions(self):
        # action 1: choose number
        # action 2: choose quantity
        # action 3: choose continuation
        return dict(nr=dict(type='int', num_values=6),
                    quant=dict(type='int', num_values=8),
                    cont=dict(type='int', num_values=3))

    def max_episode_timesteps(self):
        return 8 * 60

    def execute(self, actions):
        valid = self.validate(actions)
        if valid:
            reward = self.execute_valid_action(actions)
        else:
            reward = self.execute_invalid_action()
        if not valid or actions['cont'] > 0:
            self.next_player()
        self.roll()
        terminal = not bool([stone for stone in self.state['stone_lock']
                             if stone == 0])
        return self.state, terminal, reward

    def validate(self, action):
        nr = action['nr']
        quantity = action['quant'] + 1
        cont = action['cont']
        stone_state = self.get_stone_state()
        dice_state = self.get_dice_state()
        locked_dice = [dice[0] for dice in dice_state
                       if dice[1] == 1]
        if nr in locked_dice:
            return False
        valid_dice = [dice for dice in dice_state
                      if dice[0] == nr and dice[1] == 0]
        if len(valid_dice) < quantity:
            return False
        regenworm = [dice for dice in dice_state
                     if dice[0] == 5 and dice[1] == 1]
        if cont > 0 and not regenworm and nr != 5:
            return False
        dice_sum = sum([min(dice[0]+1, 5) for dice in dice_state
                        if dice[1] == 1]) + (min(nr+1, 5) * quantity)
        if cont == 1:
            avail_stones = [i+21 for i, stone in enumerate(stone_state)
                            if stone[0] == 0
                            and stone[1] == 0
                            and i+21 <= dice_sum]
            if not avail_stones:
                return False
        elif cont == 2:
            avail_stones = [i+21 for i, stone in enumerate(stone_state)
                            if stone[0] > 1
                            and stone[1] == 0
                            and i+21 == dice_sum]
            if not avail_stones:
                return False
        return True

    def execute_valid_action(self, action):
        reward = 1
        nr = action['nr']
        quantity = action['quant'] + 1
        cont = action['cont']
        stone_state = self.get_stone_state()
        dice_state = self.get_dice_state()
        for i, dice in enumerate(dice_state):
            if dice[0] == nr:
                self.state['dice_value'][i] = nr
                self.state['dice_lock'][i] = 1
                quantity -= 1
        dice_state = self.get_dice_state()
        dice_sum = sum([min(dice[0]+1, 5) for dice in dice_state
                        if dice[1] == 1]) + (min(nr+1, 5) * quantity)
        if cont > 0:
            last_stone = [i for i, stone in enumerate(stone_state)
                          if stone[0] == 1
                          and stone[1] == 0]
            if last_stone:
                self.state['stone_pos'][last_stone[0]] = 1
                self.state['stone_lock'][last_stone[0]] = 1
            if cont == 1:
                highest_stone = [i for i, stone in enumerate(stone_state)
                                 if stone[0] == 0
                                 and stone[1] == 0][-1]
            else:
                highest_stone = [i for i, stone in enumerate(stone_state)
                                 if stone[0] > 1
                                 and stone[1] == 0
                                 and i+21 == dice_sum][0]
            self.state['stone_pos'][highest_stone] = 1
            self.state['stone_lock'][highest_stone] = 0
            reward += floor(highest_stone/4)+1
        return reward

    def execute_invalid_action(self):
        reward = -1
        stone_state = self.get_stone_state()
        stone_stack = [i for i, stone in enumerate(stone_state)
                       if stone[0] == 1
                       and stone[1] == 0]
        if stone_stack:
            lost_stone = stone_stack[0]
            self.state['stone_pos'][lost_stone] = 0
            self.state['stone_lock'][lost_stone] = 0
            reward -= floor(lost_stone/4)+1
        highest_stone = [i for i, stone in enumerate(stone_state)
                         if stone[0] == 0
                         and stone[1] == 0]
        if highest_stone:
            self.state['stone_pos'][highest_stone[-1]] = 0
            self.state['stone_lock'][highest_stone[-1]] = 1
        return reward

    def next_player(self):
        for i, stone in enumerate(self.get_stone_state()):
            if stone[0] > 0:
                self.state['stone_pos'][i] = self.rotate(stone[0])
        self.state['dice_value'] = [0] * 8
        self.state['dice_lock'] = [0] * 8
        self.current_player = (self.current_player + 1) % self.nplayers

    def rotate(self, stone):
        return (stone % self.nplayers) + 1

    def roll(self):
        dice_state = self.get_dice_state()
        for i, dice in enumerate(dice_state):
            if dice[1] == 1:
                continue
            self.state['dice_value'][i] = randint(0,5)
            self.state['dice_lock'][i] = 0

    def get_dice_state(self):
        return [(self.state['dice_value'][i], self.state['dice_lock'][i])
                for i in range(8)]

    def get_stone_state(self):
        return [(self.state['stone_pos'][i], self.state['stone_lock'][i])
                for i in range(16)]
