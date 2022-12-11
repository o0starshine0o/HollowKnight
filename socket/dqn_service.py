# coding: utf-8
import _thread
import json
import os
import random
import socket
import sys
import time
from datetime import datetime
from itertools import chain

import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from DQN_HollowKnight.Tool.data_helper import parse_data
from DQN_HollowKnight.Tool.ui_helper import start_draw, end_draw, draw_ui


class Knight:
    def __init__(self):
        # 冗余攻击距离
        self.redundancy = 20
        # HP因子
        self.hp_scale = 10
        # distance因子
        self.distance_scale = 2
        # 基本信息
        self.hp = 9
        self.mp = 0
        # idle, running, airborne, no_input(受伤后一段时间无法接受输入)
        self.state = 'idle'
        # x, y 加速度
        self.velocity = 0, 0
        # 无敌, 被伤害后一段时间
        self.invulnerable = False
        # 正在冲刺中
        self.dashing = False
        # 正在黑冲
        self.superDashing = False
        # 正在跳跃
        self.jumping = False
        # 正在二段跳
        self.doubleJumping = False
        # 正在下落
        self.falling = False
        # 正在攻击
        self.attacking = False
        # 是否可以冲刺
        self.can_cast = False
        # 是否可以黑冲
        self.can_super_dash = False
        # 记录攻击距离(horizontal, up, down), 记录下默认值, 就不用反复计算了
        self.attack_distance = 307 - self.redundancy, 307 - self.redundancy, 273 - self.redundancy
        # 能够收获奖励的最远距离, 超过这个距离就要负reward了
        self.value_distance = 4 * self.attack_distance[0]
        # 记录每个动作的价值
        self.move_action_values = []

    def reset(self):
        self.hp = 9
        self.mp = 0
        self.state = 'idle'
        self.velocity = 0, 0
        self.invulnerable = False
        self.dashing = False
        self.superDashing = False
        self.jumping = False
        self.doubleJumping = False
        self.falling = False
        self.attacking = False
        self.can_cast = False
        self.can_super_dash = False
        self.move_action_values = []

    def get_reward(self, knight_data: dict):
        reward = self.hp_scale * (knight_data['hp'] - self.hp)
        self.hp = knight_data['hp']
        return reward

    def update_attack(self, attacks_data: list):
        x_distance, up_distance, down_distance = self.attack_distance
        if x_distance > 0 and up_distance > 0 and down_distance > 0:
            return
        for attack_data in attacks_data:
            match attack_data['name']:
                case 'Slash':
                    x = [position[0] for position in attack_data['position']]
                    x_distance = max(x) - min(x)
                case 'UpSlash':
                    y = [position[1] for position in attack_data['position']]
                    up_distance = max(y) - min(y)
                case 'DownSlash':
                    y = [position[1] for position in attack_data['position']]
                    down_distance = max(y) - min(y)
        self.attack_distance = x_distance, up_distance, down_distance
        print(f'update_attack: {self.attack_distance}')


class Boss:
    def __init__(self, name='Hornet Boss 2'):
        self.hp = 800
        self.name = name

    def reset(self, name='Hornet Boss 2'):
        self.hp = 800
        self.name = name

    def filter_boss(self, enemy: dict) -> bool:
        return enemy['name'] == self.name

    def get_reward(self, boss_data: dict):
        reward = self.hp - boss_data['hp']
        self.hp = boss_data['hp']
        return reward


class Pool:

    def __init__(self, save_path: str = None, size: int = 10 * 60 * 60):
        self.size = size
        self.names = '/states.npy', '/moves.npy', '/actions.npy', '/move_rewards.npy', '/action_rewards.npy'
        if self._file_exist(save_path):
            self.states, self.moves, self.actions, self.move_rewards, self.action_rewards = \
                tuple(np.load(save_path + name) for name in self.names)
            self.current, self.count, size = tuple(np.load(save_path + '/meta.npy'))
            print(f'load pool:[{self.current}|{self.count}|{size}]', save_path)
        else:
            self.states = np.empty((self.size, 32))
            self.moves = np.empty(self.size, dtype=int)
            self.actions = np.empty(self.size, dtype=int)
            self.move_rewards = np.zeros(self.size)
            self.action_rewards = np.zeros(self.size)
            self.current = 0
            self.count = 0
        self.data = (self.states, self.moves, self.actions, self.move_rewards, self.action_rewards)

    def record(self, state: np.ndarray, move: int, action: int, move_reward: float, action_reward: float):
        self.states[self.current] = state
        self.moves[self.current] = move
        self.actions[self.current] = action
        # reward 为上一步的action的奖励, 所以往上推一步
        self.move_rewards[self.current - 1] = move_reward
        self.action_rewards[self.current - 1] = action_reward
        self.current += 1
        self.count = max(self.count, self.current)
        self.current %= self.size

    def recall(self, size: int) -> tuple | None:
        """
        Returns:
            states: 选取的状态S0
            moves: 此状态下选取的移动A0
            actions: 此状态下选取的动作A0
            move_rewards: 选取移动的奖励R0
            action_rewards: 选取动作的奖励R0
            next_states: 到达的下一个状态S1
        """
        if self.count < size:
            print('size < count', 'not enough data')
            return
        indices = np.array(random.sample(range(self.count), size))
        next_indices = (indices + 1) % self.count
        all_data = (self.states, self.moves, self.actions, self.move_rewards, self.action_rewards)
        return tuple(data[indices] for data in all_data) + (self.states[next_indices],)

    def save(self, save_path: str):
        if save_path and os.path.exists(save_path):
            list(map(lambda name, data: np.save(save_path + name, data), self.names, self.data))
            np.save(save_path + '/meta.npy', np.array([self.current, self.count, self.size]))
            print(f'save pool: {save_path} [{self.current}|{self.count}|{self.size}]')

    def _file_exist(self, save_path: str = None) -> bool:
        return save_path and all(map(lambda file: os.path.exists(save_path + file), self.names + ('/meta.npy',)))


class Agent:

    def __init__(self, save_path: str = None, input_size=32, gamma=0.99, learning_rate=0.001, batch_size=32):
        if save_path and os.path.exists(save_path):
            self.model = tf.keras.models.load_model(save_path + '/model.tf')
            print(f'load model:', save_path + '/model.tf')
        else:
            self.model = self.__build_deep_q_network__(Input(input_size), learning_rate)
        # 程序启动就开始学习, 生命不停, 学习不止
        self.learn_thread = _thread.start_new_thread(self._learn, (gamma, batch_size))
        self.move_loss = sys.maxsize
        self.action_loss = sys.maxsize

    def __build_deep_q_network__(self, input_layer: Input, learning_rate: float = 0.001):
        hidden = Dense(256, relu)(input_layer)
        hidden = Dense(2048, relu)(hidden)
        flatten = Flatten()(hidden)
        move = Dense(128, relu)(flatten)
        action = Dense(128, relu)(flatten)
        move_output_layer = Dense(len(game.moves), softmax)(move)
        action_output_layer = Dense(len(game.actions), softmax)(action)
        model = Model(input_layer, [move_output_layer, action_output_layer])
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
        return model

    def get_action(self, state: np.ndarray, fight_count=0):
        """
        返回一个action的索引
        """
        # 经验池比较小, 采取随机操作, 增大经验池
        move_random = agent.move_loss > 20
        action_random = agent.action_loss > 20
        moves, actions = tuple(prediction.numpy() for prediction in self.__get_prediction__(state))
        move_index = random.randint(0, game.len_moves - 1) if move_random else moves.argmax()
        action_index = random.randint(0, game.len_actions - 1) if action_random else actions.argmax()
        knight.move_action_values.append(list(map(float, np.concatenate((moves, actions), 1)[0])))
        # 记录每一个动作的价值
        with tf.name_scope(f"fight_count[{fight_count}]"):
            action_value = dict(zip(game.move_action_names, knight.move_action_values[-1]))
            tf.summary.text('action values', json.dumps(action_value), len(knight.move_action_values))
        writer.flush()

        return move_index, action_index, move_random, action_random

    def _learn(self, gamma=0.99, batch_size=32, empty_sleep_time=10):
        """
        这是一个线程函数, 用到write需要单独再创建
        """
        step = 0
        start = datetime.now()
        try:
            with writer.as_default():
                while True:
                    data = pool.recall(batch_size)
                    if not data:
                        # 避免无数据时空转
                        time.sleep(empty_sleep_time)
                        continue

                    # 核心梯度下降过程
                    move_loss, action_loss = self._learn_kernel(*data, gamma=gamma)
                    self.move_loss, self.action_loss = move_loss.numpy(), action_loss.numpy()

                    step += 1

                    # 保存损失信息
                    if step % 10 == 0:
                        tf.summary.scalar('move_loss', self.move_loss, step)
                        tf.summary.scalar('action_loss', self.action_loss, step)
                        writer.flush()

                    # 输出一些信息表明线程正常运行
                    if step % 100 == 0:
                        # 不解释
                        print(f'good good study: {datetime.now() - start}, '
                              f'day day up {self.move_loss}, {self.action_loss}')
                        start = datetime.now()
        except KeyboardInterrupt:
            writer.close()

    def _learn_kernel(self, states: np.ndarray, moves: np.ndarray, actions: np.ndarray, move_rewards: np.ndarray,
                      action_rewards: np.ndarray, next_states: np.ndarray, gamma=0.99) -> (tf.Tensor, tf.Tensor):
        # 1, 根据当前网络, 计算next_states下所有action的value, 并且选取价值最大的那个动作的值, 作为next_q_max
        # next_moves_value_max = max(Q(S_1, W)), shape: (32, )
        # next_actions_value_max = max(Q(S_1, W)), shape: (32,)
        next_moves_value_max, next_actions_value_max = \
            tuple(prediction.numpy().max(axis=1) for prediction in self.__get_prediction__(next_states))
        # 2, 添加gamma, reward作为target_q
        # target_move_q = R_0 + gamma * next_q_max, shape: (32, )
        target_move_q = move_rewards + gamma * next_moves_value_max
        # target_action_q = R_0 + gamma * next_q_max, shape: (32, )
        target_action_q = action_rewards + gamma * next_actions_value_max
        with tf.GradientTape() as tape:
            # 3, 根据当前网络, 重新计算states下所有moves和actions的value
            # move_values = Q(S0, W), shape: (32, 3)
            # action_values = Q(S0, W), shape: (32, 5)
            move_values, action_values = tuple(self.__get_prediction__(states))
            # 当时选中的move和action, 转换为one_hot, shape: (32, 3)和(32, 5)
            one_hot_moves = tf.keras.utils.to_categorical(moves, game.len_moves)
            one_hot_actions = tf.keras.utils.to_categorical(actions, game.len_actions)
            # 根据当时选择的move和action(可能是随机选择的, 不一定是argmax), 和现在的网络参数, 计算q值, shape: (32, )
            move_q = tf.reduce_sum(tf.multiply(move_values, one_hot_moves), axis=1)
            action_q = tf.reduce_sum(tf.multiply(action_values, one_hot_actions), axis=1)
            # 4, 使用2者的MSE作为loss(差值小于1), 否则使用绝对值(差值大于1)
            # loss = (target_q - q)^2
            move_loss = self.model.loss(target_move_q, move_q)
            action_loss = self.model.loss(target_action_q, action_q)
            # 5, 计算梯度, 结果是一个list, 记录了每一层, 每一个cell需要下降的梯度
            # 多个loss合并, 一同梯度下降
            gradients = tape.gradient(move_loss + action_loss, self.model.trainable_variables)
            # 梯度不大, 没必要裁剪
            # gradients = [tf.clip_by_norm(gradient, 15) for gradient in gradients]
        # 6, 反向传播, 更新model的参数
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return move_loss, action_loss

    def save(self, save_path: str = None):
        if save_path and os.path.exists(save_path):
            self.model.save(save_path + '/model.tf')
            print('save model:', save_path + '/model.tf')

    @tf.function
    def __get_prediction__(self, state: np.ndarray) -> list[tf.Tensor]:
        """
        Return: list: [shape(1, 3), shape(1, 5)]
        """
        return self.model(state)


class Game:
    """
    负责与游戏交互
    """

    def __init__(self):
        self.width, self.height = 2560, 1378
        self.moves = [
            ('idle', None),
            ('left', self._left),
            ('right', self._right),
        ]
        self.actions = [
            ('idle', None),
            ('attack', 'j'),
            ('jump', 'k'),
            ('up_attack', ['w', 'j']),
            ('down_attack', ['s', 'j'])
        ]
        self.len_actions = len(self.actions)
        self.len_moves = len(self.moves)
        self.move_names = [move[0] for move in self.moves]
        self.action_names = [action[0] for action in self.actions]
        self.move_action_names = self.move_names + self.action_names

    def step(self, move: str | list[str] | tuple, action: str | list[str] | tuple):
        self._step(move)
        self._step(action)

    def _step(self, step: str | list[str] | tuple):
        match step:
            case ('left', function) | ('right', function):
                function()
            case tuple():
                self._step(step[1])
            case list():
                pyautogui.hotkey(*step)
            case str():
                pyautogui.press(step)

    def challenge(self):
        _thread.start_new_thread(self._challenge, ())

    def _challenge(self):
        # 取消目前的长按效果
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')
        # 允许其他操作来打断无限循环
        time.sleep(5)
        # 给一个动作, 让Knight醒来
        pyautogui.press('a')
        time.sleep(2)
        # 走到雕像面前
        pyautogui.press('a')
        time.sleep(0.1)
        pyautogui.press('a')
        time.sleep(0.1)
        # 打开挑战面板
        pyautogui.press('w')
        time.sleep(1)
        # 选择挑战开始
        pyautogui.press('k')
        time.sleep(1)

    def _left(self):
        pyautogui.keyUp('d')
        pyautogui.keyDown('a')

    def _right(self):
        pyautogui.keyUp('a')
        pyautogui.keyDown('d')


class Turing:
    _time_format = '%m/%d/%Y %I:%M:%S %p.%f'

    def __init__(self):
        # 配置socket
        self.socket_service = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定socket文件
        self.socket_service.bind(("127.0.0.1", 9203))
        # 如果队列满了, 请求会被拒绝, 这里多创建些, 防止队列满了
        self.socket_service.listen(10)
        # 记录进入BOSS的时间, 方便计算与BOSS战斗的时间
        self.boss_start = datetime.now()
        # 记录本次turing与boss对战的次数
        self.fight_count = 0
        # 记录下HollowKnight发送状态到收到动作的时间延迟
        self.delay_time = []
        # 不解释
        print("Make AI Great Again")

    def start(self):
        # 使用一个死循环, 不断的接受客户端的消息
        try:
            with writer.as_default():
                while True:
                    self._game_start()
        except KeyboardInterrupt:
            print('end')
            writer.close()

    def save(self):
        # Create the folder for saving the agent
        save_file_path = get_save_path('save/')
        # 保存接下来的数据
        agent.save(save_file_path)
        pool.save(save_file_path)

    def _game_start(self):
        # 接收其他进程连接
        connection, address = self.socket_service.accept()
        # 当前场景
        scene = ""

        while True:
            try:
                # 收到的原始数据
                origin_data = connection.recv(2048)
                # 如果没有数据, 就断开连接
                if not origin_data:
                    break
                # 处理当前帧
                scene = self._frame_current(origin_data, scene)
                # 反馈给客户端, 保证每次接受的都是一帧的数据
                move_index, action_index = tuple(int(data[pool.current - 1]) for data in (pool.moves, pool.actions))
                connection.send(str.encode(json.dumps((move_index, action_index))))
            except ConnectionResetError as exception:
                print(exception)

        # 保存当前记录
        self.save()

    def _frame_current(self, origin_data: bytes, pre_scene: str):
        # 记录日志
        receive_time = datetime.now()
        # 保存为字典类型
        json_data = parse_data(origin_data)
        if json_data is None:
            return
        # 发送时间
        send_time = datetime.strptime(json_data['time'], self._time_format)
        # 延迟的时间
        self.delay_time.append(receive_time - send_time)

        # 根据不同场景, 进入到不同的任务
        scene = json_data['scene']
        collider = json_data['collider']
        if collider['Attacks']:
            knight.update_attack(collider['Attacks'])
        match pre_scene:
            case 'GG_Workshop':
                match scene:
                    case 'GG_Hornet_2':
                        self._before_boss(receive_time)
            case 'GG_Hornet_2':
                match scene:
                    case 'GG_Hornet_2':
                        _knight = collider['Knight']
                        knight_reward = knight.get_reward(_knight)
                        _enemies = collider['Enemies']
                        boss_reward = boss.get_reward(next(filter(boss.filter_boss, _enemies))) if _enemies else 0
                        move_reward = self._get_move_reward(_knight, _enemies) if _enemies else 0
                        hp_reward = knight_reward + boss_reward
                        # draw_ui(collider['Knight'], collider['Enemies'], collider['Attacks'])

                        knight_points = _knight['position']
                        enemy_points = list(chain(*[enemy['position'] for enemy in _enemies]))

                        state, move_index, action_index, move_random, action_random = \
                            self._fight_boss(knight_points, enemy_points)
                        pool.record(state, move_index, action_index, move_reward, 0)

                        if move_reward or hp_reward:
                            print(receive_time, 'Turing',
                                  '[random]:' if move_random else '[prediction]:',
                                  'move', game.move_names[move_index], 'get', move_reward,
                                  '[random]:' if action_random else '[prediction]:',
                                  'action', game.action_names[action_index], 'get', hp_reward,
                                  )
                    case 'GG_Workshop':
                        self._end_boss(receive_time)
        return scene

    def _before_boss(self, boss_start: datetime):
        """ 进入BOSS场景, 需要初始化一些操作
        """
        self.boss_start = boss_start
        self.delay_time = []
        knight.reset()
        boss.reset()
        start_draw()

    def _get_move_reward(self, knight_data: dict, enemies_data: list):
        boss_data = next(filter(boss.filter_boss, enemies_data))
        boss_x_list = [position[0] for position in boss_data['position']]
        boss_y_list = [position[1] for position in boss_data['position']]
        knight_x_list = [position[0] for position in knight_data['position']]
        knight_y_list = [position[1] for position in knight_data['position']]
        # 两两相减, 得出最小的距离
        x_distance = min([abs(boss_x - knight_x) for knight_x in knight_x_list for boss_x in boss_x_list])
        y_distance = min([abs(boss_y - knight_y) for knight_y in knight_y_list for boss_y in boss_y_list])
        # 计算奖励
        attack_x, attack_up, attack_down = knight.attack_distance
        # 攻击范围内和攻击范围外使用2套奖励逻辑
        if x_distance <= attack_x:
            # 使用线性函数y = kx + b
            b = -knight.hp_scale
            k = (knight.distance_scale + knight.hp_scale) / attack_x
            return k * x_distance + b
        else:
            # 使用另外一个线性函数 (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)
            # y = (y2 - y1)*(x - x1)/(x2 - x1) + y1
            return (-knight.distance_scale) * (x_distance - attack_x) / (knight.value_distance - attack_x) \
                   + knight.distance_scale

    def _fight_boss(self, knight_points: list[list[int]], enemies_points: list[list[int]]):
        """
        knight的坐标, 长度为2, [[left, top], [right, bottom]]
        enemies的坐标, 目前有5个, 先固定分配位置, 第二维长度为4,
        [[left, top], [right, bottom]], [[left, top], [right, bottom]]]
        """
        # 把list转换为ndarray
        positions = [[point[0] / game.width, point[1] / game.height] for point in knight_points + enemies_points]
        # 调整形状: (n,), 最多保留32个
        state_flatten = np.array(positions).flatten()[:32]
        # (32,), 最多少保留32个
        state_fixed = np.pad(state_flatten, (0, max(0, 32 - len(state_flatten))))
        # (1, 32)
        state_reshape = np.reshape(state_fixed, (1, 32))
        # 把得到的状态给DQN, 拿到action
        move_index, action_index, move_random, action_random = agent.get_action(state_reshape, self.fight_count)
        # 把dqn计算得到的action给游戏
        # game.step(game.moves[move_index], game.actions[0])

        return state_fixed, move_index, action_index, move_random, action_random

    def _end_boss(self, end_time: datetime):
        """ 离开BOSS场景, 需要统计一些操作
        """
        # 本场统计
        print("Fight boss within:", (end_time - self.boss_start).seconds, "s")
        print("Take actions:", len(self.delay_time))
        if len(self.delay_time) > 0:
            average_time_delay = np.mean(list(map(lambda delay: delay.total_seconds(), self.delay_time)))
            print("Average time delay:", average_time_delay * 1000, "ms")
        # 记录下boss的血量
        tf.summary.scalar('BOSS HP', boss.hp, self.fight_count)
        # 统计每个动作的使用次数
        with tf.name_scope(f"fight_count[{self.fight_count}]"):
            move_frequency = [0] * game.len_moves
            action_frequency = [0] * game.len_actions
            for move_action_value in knight.move_action_values:
                value = np.array(move_action_value)
                move_frequency[value[:game.len_moves].argmax()] += 1
                action_frequency[value[game.len_moves:].argmax()] += 1
            move_frequency = json.dumps(dict(zip(game.move_names, move_frequency)))
            action_frequency = json.dumps(dict(zip(game.action_names, action_frequency)))
            print(f'move frequency: {move_frequency}')
            print(f'action frequency: {action_frequency}')
            tf.summary.text('move action frequency', json.dumps((move_frequency, action_frequency)),
                            len(knight.move_action_values) + 1)
        writer.flush()
        self.fight_count += 1

        # 准备下一场挑战
        # game.challenge()
        end_draw()


def get_save_path(prefix, time_format='%m-%d_%H:%M:%S'):
    save_file_path = prefix + start_time.strftime(time_format)
    if not os.path.isdir(save_file_path):
        os.makedirs(save_file_path)
    return save_file_path


if __name__ == '__main__':
    start_time = datetime.now()
    writer = tf.summary.create_file_writer(get_save_path('log/'))
    game = Game()
    knight = Knight()
    boss = Boss()
    pool = Pool('save/' + sys.argv[1])
    agent = Agent('save/' + sys.argv[1])
    Turing().start()
