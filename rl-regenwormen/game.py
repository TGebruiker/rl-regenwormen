from tensorforce.environments import Environment
from math import floor
from random import randint


class Game(Environment):
    def __init__(self, nplayers=4):
        super().__init__()
        self.nplayers = nplayers
        self.state = self.reset()
        self.terminal = False

    def reset(self):
        self.current_player = 0
        self.state = {'stone_pos': [0] * 16,
                      'stone_lock': [0] * 16,
                      'dice_lock': [0] * 6,
                      'dice_free': [0] * 6}
        return self.state

    def states(self):
        # 15 stones and 8 dice
        return dict(stone_pos=dict(type='int', shape=(16,), num_values=self.nplayers+1),
                    stone_lock=dict(type='int', shape=(16,), num_values=2),
                    dice_lock=dict(type='int', shape=(6,), num_values=9),
                    dice_free=dict(type='int', shape=(6,), num_values=9))

    def actions(self):
        # action 1: choose number
        # action 2: choose quantity
        # action 3: choose continuation
        return dict(nr=dict(type='int', num_values=6),
                    quant=dict(type='int', num_values=8),
                    cont=dict(type='int', num_values=3))

    def max_episode_timesteps(self):
        return 25

    def execute(self, actions):
        valid = self.validate(actions)
        if valid:
            reward = self.execute_valid_action(actions)
        else:
            reward = 0
            if self.check_possible_move():
                reward = -1
            reward -= self.execute_invalid_action()

        terminal = sum(self.state['dice_free']) == 0 or actions['cont'] > 0
        if terminal:
            self.next_player()

        if not bool([stone for stone in self.state['stone_lock']
                     if stone == 0]):
            self.reset()
        self.roll()
        
        action_mask = self.generate_mask()
        return {**self.state, **action_mask}, terminal, reward

    def validate(self, action):
        nr = action['nr']
        quantity = action['quant'] + 1
        cont = action['cont']
        stone_state = self.get_stone_state()
        dice_locked, dice_free = self.state['dice_lock'], self.state['dice_free']
        locked_dice = set([i for i, v in enumerate(dice_locked)
                           if v > 0])
        if nr in locked_dice:
            return False
        if dice_free[nr] < quantity:
            return False
        regenworm = dice_free[5] > 0
        if cont > 0 and not regenworm and nr != 5:
            return False
        dice_sum = sum(dice_locked) + (nr + 1) * quantity
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

    def check_possible_move(self):
        return 0 in self.state['dice_lock']

    def execute_valid_action(self, action):
        reward = 0
        nr = action['nr']
        quantity = action['quant'] + 1
        cont = action['cont']
        stone_state = self.get_stone_state()
        self.state['dice_lock'][nr] = quantity
        self.state['dice_free'][nr] -= quantity
        dice_locked = self.state['dice_lock']
        dice_sum = sum(dice_locked)
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
            reward += (((highest_stone+1)/16) + 1) * cont
        else:
            reward += (((nr+1) * quantity) / 48)
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
            reward -= ((lost_stone + 1)/16) + 1
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
        self.state['dice_free'] = [0] * 6
        self.state['dice_lock'] = [0] * 6
        self.current_player = (self.current_player + 1) % self.nplayers

    def rotate(self, stone):
        return (stone % self.nplayers) + 1

    def roll(self):
        self.state['dice_free'] = [0] * 6
        dice_lock = self.state['dice_lock']
        for _ in range(8 - sum(dice_lock)):
            value = randint(0, 5)
            self.state['dice_free'][value] += 1

    def get_stone_state(self):
        return [(self.state['stone_pos'][i], self.state['stone_lock'][i])
                for i in range(16)]

    def end_round(self):
        stone_state = self.get_stone_state()
        player_stone_sums = [0 for _ in range(self.nplayers)]
        for i, stone in enumerate(stone_state):
            if stone[0] > 0:
                player_stone_sums[stone[0] - 1] += floor(i/4)+1
        if player_stone_sums[0] == max(player_stone_sums) and player_stone_sums[0] > 0:
            return 100
        else:
            return 0

    def generate_mask(self):
        dice_locked = self.state['dice_lock']
        dice_free = self.state['dice_free']
        nr_mask = [False if dice > 0 or dice_free[i] == 0
                   else True for i, dice in enumerate(dice_locked)]
        max_quant = max(self.state['dice_free']) - 1
        quant_mask = [True] * max_quant + [False] * (8-max_quant)
        dice_sum_combs = [sum(dice_locked) + nr+1 * quant for nr, max_quant in enumerate(dice_free)
                          for quant in range(1, max_quant+2)]
        stone_pos = self.state['stone_pos']
        stone_lock = self.state['stone_lock']
        stones_on_table = [50] + [stone+21 for stone, pos in enumerate(stone_pos)
                                  if stone_lock[stone] == 0 and pos == 0]
        cont_1 = max(dice_sum_combs) >= min(stones_on_table)
        stones_to_steal = [stone+21 for stone, pos in enumerate(stone_pos)
                           if stone_lock[stone] == 0 and pos > 1]
        total = dice_sum_combs + stones_to_steal
        cont_2 = len(total) > len(set(total))
        cont_mask = [True, cont_1, cont_2]
        return dict(nr_mask=nr_mask, quant_mask=quant_mask, cont_mask=cont_mask)
