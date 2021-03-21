from tensorforce.environments import Environment
from math import floor
from random import randint


class Game(Environment):
    def __init__(self, nplayers=4, show=False):
        super().__init__()
        self.show = show
        self.nplayers = nplayers
        self.state = self.reset()
        self.terminal = False

    def reset(self):
        self.current_player = 0
        self.state = {'stone_pos': [0] * 16,
                      'stone_lock': [0] * 16,
                      'dice_lock': [0] * 6,
                      'dice_free': [0] * 6,
                      'nr_action': 0,
                      'quant_action': 0}
        self.roll()
        mask = self.generate_nr_mask()
        self.terminal = False
        return {**self.state, "action_mask": mask}

    def states(self):
        # 15 stones and 8 dice
        return dict(stone_pos=dict(type='int', shape=(16,), num_values=self.nplayers+1),
                    stone_lock=dict(type='int', shape=(16,), num_values=2),
                    dice_lock=dict(type='int', shape=(6,), num_values=9),
                    dice_free=dict(type='int', shape=(6,), num_values=9),
                    nr_action=dict(type='int', num_values=7),
                    quant_action=dict(type='int', num_values=9))

    def max_episode_timesteps(self):
        return 25

    def execute_nr(self, action):
        self.nr_action = action
        dice_free = self.state['dice_free']
        quant = dice_free[action]
        mask = [True] * quant + [False] * (8-quant)
        self.state["nr_action"] = action + 1
        return {**self.state, "action_mask": mask}

    def execute_quant(self, action):
        self.quant_action = action + 1
        dice_locked = self.state['dice_lock']
        dice_sum = sum(dice_locked)
        extra_sum = (self.nr_action + 1) * (action + 1)
        total = dice_sum + extra_sum
        stone_pos = self.state['stone_pos']
        stone_lock = self.state['stone_lock']
        stones_on_table = [50] + [stone+21 for stone, pos in enumerate(stone_pos)
                                  if stone_lock[stone] == 0 and pos == 0]
        cont_1 = total >= min(stones_on_table)
        stones_to_steal = [stone+21 for stone, pos in enumerate(stone_pos)
                           if stone_lock[stone] == 0 and pos > 1]
        cont_2 = total in stones_to_steal
        mask = [True, cont_1, cont_2]
        self.state["quant_action"] = action + 1
        return {**self.state, "action_mask": mask}

    def execute(self, cont_action):
        terminal = False
        reward = self.execute_valid_action(cont_action)
        self.roll()
        nr_mask = self.generate_nr_mask()
        if not any(nr_mask) or cont_action > 0:
            if cont_action == 0:
                reward -= 0.5
                self.flip_stone()
            terminal = True
            self.next_player()
            nr_mask = self.generate_nr_mask()
        if not bool([stone for stone in self.state['stone_lock']
                     if stone == 0]):
            self.terminal = True
            
        self.state["nr_action"] = 0
        self.state["quant_action"] = 0
        return {**self.state, "action_mask": nr_mask}, terminal, reward

    def flip_stone(self):
        stone_state = self.get_stone_state()
        stone_stack = [i for i, stone in enumerate(stone_state)
                       if stone[0] == 1
                       and stone[1] == 0]
        if stone_stack:
            lost_stone = stone_stack[0]
            self.state['stone_pos'][lost_stone] = 0
            self.state['stone_lock'][lost_stone] = 0
        highest_stone = [i for i, stone in enumerate(stone_state)
                         if stone[0] == 0
                         and stone[1] == 0]
        if highest_stone:
            self.state['stone_pos'][highest_stone[-1]] = 0
            self.state['stone_lock'][highest_stone[-1]] = 1

    def execute_valid_action(self, action):
        reward = 0
        nr = self.nr_action
        quantity = self.quant_action
        cont = action
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
            reward += (((highest_stone+1)/16) + 1) * (cont + 1) + 10
        #else:
        #    reward += (((nr+1) * quantity) / 48)
        return reward

    def next_player(self):
        for i, stone in enumerate(self.get_stone_state()):
            if stone[0] > 0:
                self.state['stone_pos'][i] = self.rotate(stone[0])
        self.state['dice_free'] = [0] * 6
        self.state['dice_lock'] = [0] * 6
        self.current_player = (self.current_player + 1) % self.nplayers
        self.roll()

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

    def generate_nr_mask(self):
        dice_locked = self.state['dice_lock']
        dice_free = self.state['dice_free']
        nr_mask = [False if dice > 0 or dice_free[i] == 0
                   else True for i, dice in enumerate(dice_locked)]
        return nr_mask
