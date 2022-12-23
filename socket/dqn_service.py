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
from random import choices

import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.python.keras.activations import relu, softmax
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from DQN_HollowKnight.Tool.data_helper import parse_data
from DQN_HollowKnight.Tool.ui_helper import start_draw, end_draw


class Knight:
    def __init__(self):
        # 冗余攻击距离
        self.redundancy = 20
        # HP因子, 因为knight的血量变化为1, 需要对其缩放
        self.hp_scale = 10
        # x轴奖励
        self.x_reward = 2
        # y轴奖励
        self.y_reward = 2
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
        self.canCast = False
        # 是否可以黑冲
        self.canSuperDash = False
        # 是否接触到墙壁
        self.touchingWall = False
        # 是否面朝右方
        self.facingRight = True
        # 记录攻击距离(horizontal, up, down), 记录下默认值, 就不用反复计算了
        self.attack_distance = 307 - self.redundancy, 307 - self.redundancy, 273 - self.redundancy
        # 能够收获奖励的最远距离, 超过这个距离就要负reward了
        self.x_distance = 4 * self.attack_distance[0]
        self.y_distance = 4 * self.attack_distance[2]
        # 记录每个动作的价值
        self.move_values = []
        self.action_values = []

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
        self.touchingWall = False
        self.facingRight = True
        self.canCast = False
        self.canSuperDash = False
        self.move_values = []
        self.action_values = []

    def update(self, _knight: dict):
        self.hp = _knight['hp']
        self.mp = _knight['mp']
        self.state = _knight['state']
        self.velocity = _knight['velocity']
        self.invulnerable = _knight['invulnerable']
        self.dashing = _knight['dashing']
        self.superDashing = _knight['superDashing']
        self.jumping = _knight['jumping']
        self.doubleJumping = _knight['doubleJumping']
        self.falling = _knight['falling']
        self.attacking = _knight['attacking']
        self.canCast = _knight['canCast']
        self.canSuperDash = _knight['canSuperDash']
        self.touchingWall = _knight['touchingWall']
        self.facingRight = _knight['facingRight']

    def get_reward(self, knight_data: dict):
        return self.hp_scale * (knight_data['hp'] - self.hp)

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
    def __init__(self, name='Hornet Boss 1'):
        self.hp = 900
        self.name = name

    def reset(self, name='Hornet Boss 1'):
        self.hp = 900
        self.name = name

    def filter_boss(self, enemy: dict) -> bool:
        return enemy['name'] == self.name

    def get_reward(self, boss_data: dict):
        reward = self.hp - boss_data['hp']
        self.hp = boss_data['hp']
        return reward


class Pool:
    __range = 30

    def __init__(self, save_path: str = None, size: int = 10 * 60 * 60):
        self.size = size
        self.names = '/states.npy', '/moves.npy', '/actions.npy', '/move_rewards.npy', '/action_rewards.npy', \
                     '/losses.npy'

        if self._file_exist(save_path):
            self.states, self.moves, self.actions, self.move_rewards, self.action_rewards, self.losses = \
                tuple(np.load(save_path + name) for name in self.names)
            self.current, self.count, size = tuple(np.load(save_path + '/meta.npy'))
            print(f'load pool:[{self.current}|{self.count}|{size}]', save_path)
        else:
            self.states = np.empty((self.size, 32))
            self.moves = np.empty(self.size, dtype=int)
            self.actions = np.empty(self.size, dtype=int)
            self.move_rewards = np.zeros(self.size)
            self.action_rewards = np.zeros(self.size)
            self.losses = np.ones(self.size)
            self.current = 0
            self.count = 0
        self.data = (self.states, self.moves, self.actions, self.move_rewards, self.action_rewards, self.losses)

    def record(self, state: np.ndarray, move: int, action: int, move_reward: float, action_reward: float):
        self.states[self.current] = state
        self.moves[self.current] = move
        self.actions[self.current] = action
        # reward 为上一步的action的奖励, 所以往上推一步
        self.move_rewards[self.current - 1] = move_reward
        self.action_rewards[self.current - 1] = action_reward

        # 一旦获取到奖励, 把奖励叠加到之前的N个动作上去, 尝试解决稀疏奖励问题
        if move_reward != 0:
            for before in range(1, self.__range):
                self.move_rewards[self.current - 1 - before] += move_reward * (self.__range - before) / self.__range
        if action_reward != 0:
            for before in range(1, self.__range):
                self.action_rewards[self.current - 1 - before] += action_reward * (self.__range - before) / self.__range

        self.current += 1
        self.count = max(self.count, self.current)
        self.current %= self.size

    def recall(self, size: int, with_weight=False) -> tuple | None:
        """
        Returns:
            indices: 选取的索引
            states: 选取的状态S0
            moves: 此状态下选取的移动A0
            actions: 此状态下选取的动作A0
            move_rewards: 选取移动的奖励R0
            action_rewards: 选取动作的奖励R0
            loss: 当前状态对应的损失L0
            next_states: 到达的下一个状态S1
        """
        if self.count < size:
            print('size < count', 'not enough data')
            return
        # 当前批次选中的索引
        indices = choices(range(self.count), self.losses[:self.count], k=size) if with_weight \
            else choices(range(self.count), k=size)
        indices = np.array(indices)
        # 当前批次选中的索引对应的下一个索引
        next_indices = (indices + 1) % self.count
        # 返回所有选中的数据
        return (indices,) + tuple(data[indices] for data in self.data) + (self.states[next_indices],)

    def update_loss(self, indices: np.ndarray, move_action_loss: np.ndarray):
        for index, loss in zip(indices, move_action_loss):
            self.losses[index] = loss

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
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE))
        return model

    def get_action(self, state: np.ndarray, fight_count=0):
        """
        返回一个action的索引
        """
        # 经验池比较小, 采取随机操作, 增大经验池
        # 即便经验池够了, 还是需要一定的探索性
        move_random = pool.current < 1000 or random.random() < 0.05
        action_random = pool.current < 1000 or random.random() < 0.05

        move_values, action_values = tuple(prediction.numpy() for prediction in self.__get_prediction__(state))

        move_index = self._get_random_move_index() if move_random else move_values.argmax()
        action_index = random.randint(0, game.len_actions - 1) if action_random else action_values.argmax()

        knight.move_values.append(tuple(map(float, move_values[0])))
        knight.action_values.append(tuple(map(float, action_values[0])))
        # 记录每一个动作的价值
        with tf.name_scope(f"fight_count[{fight_count}]"):
            move_value = dict(zip(game.move_names, knight.move_values[-1]))
            action_value = dict(zip(game.action_names, knight.action_values[-1]))
            tf.summary.text('move action values', json.dumps((move_value, action_value)), len(knight.move_values))
        writer.flush()

        return move_index, action_index, move_random, action_random

    def _get_random_move_index(self) -> int:
        if knight.touchingWall:
            return 1 if knight.facingRight else 2
        else:
            return 2 if knight.facingRight else 1

    def _learn(self, gamma=0.99, batch_size=32, empty_sleep_time=10):
        """
        这是一个线程函数, 用到write需要单独再创建
        """
        step = 0
        start = datetime.now()
        try:
            with writer.as_default():
                while True:
                    data = pool.recall(batch_size, True)
                    if not data:
                        # 避免无数据时空转
                        time.sleep(empty_sleep_time)
                        continue

                    indices, states, moves, actions, move_rewards, action_rewards, _, next_states = data
                    # 核心批梯度下降过程
                    move_loss, action_loss = self._learn_kernel(states, moves, actions, move_rewards, action_rewards,
                                                                next_states, gamma)
                    # 更新loss
                    pool.update_loss(indices, (move_loss + action_loss).numpy())
                    # 保存批loss的均值
                    self.move_loss = tf.reduce_mean(move_loss).numpy()
                    self.action_loss = tf.reduce_mean(action_loss).numpy()

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
            # 把入参由(32,)改为(1, 32), 方便保留每一个loss
            move_loss = self.model.loss(tf.reshape(target_move_q, [-1, 1]), tf.reshape(move_q, [-1, 1]))
            action_loss = self.model.loss(tf.reshape(target_action_q, [-1, 1]), tf.reshape(action_q, [-1, 1]))
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
        # 屏幕宽高, 用于把绝对坐标转换成float
        self.width, self.height = 2560, 1378
        # 墙坐标, 用于计算move_reward, 避免一直撞墙
        self.wall = [293, 2260]
        # 用于计算knight与wall的距离后给与的奖励
        self.wall_reward = -2
        # 距离墙多近开始判断, 认为估计用攻击距离的一半吧
        self.wall_distance = 140
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
            ('down_attack', ['s', 'j']),
            ('dash', 'l'),
            ('ball', 'o'),
            ('up_ball', ['w', 'o']),
            ('down_ball', ['s', 'o']),
        ]
        self.len_actions = len(self.actions)
        self.len_moves = len(self.moves)
        self.move_names = [move[0] for move in self.moves]
        self.action_names = [action[0] for action in self.actions]
        self.move_action_names = self.move_names + self.action_names

    def update(self, _knight: dict):
        if _knight['touchingWall']:
            point_x = [point[0] for point in _knight['position']]
            if _knight['facingRight']:
                self.wall[1] = point_x[1]
            else:
                self.wall[0] = point_x[0]

    def get_wall_reward(self, _knight: dict) -> float:
        knight_x_list = [position[0] for position in _knight['position']]
        distance = min([abs(wall_x - knight_x) for knight_x in knight_x_list for wall_x in game.wall])
        reward = (self.wall_distance * self.wall_reward - self.wall_reward * distance) / self.wall_distance
        return max(min(reward, 0), self.wall_reward)

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
        # # 记录下HollowKnight发送状态到收到动作的时间延迟
        # self.delay_time = []
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
        # # 发送时间
        # send_time = datetime.strptime(json_data['time'], self._time_format)
        # # 延迟的时间
        # self.delay_time.append(receive_time - send_time)

        # 根据不同场景, 进入到不同的任务
        scene = json_data['scene']
        collider = json_data['collider']
        if collider['Attacks']:
            knight.update_attack(collider['Attacks'])
        match pre_scene:
            case 'GG_Workshop':
                match scene:
                    case 'GG_Hornet_1':
                        self._before_boss(receive_time)
            case 'GG_Hornet_1':
                match scene:
                    case 'GG_Hornet_1':
                        # knight奖励与是否掉血相关
                        # 取值范围[0|-10]
                        _knight = collider['Knight']
                        knight_reward = knight.get_reward(_knight)
                        knight.update(_knight)

                        # boss奖励与boss是否掉血相关
                        _enemies = collider['Enemies']
                        boss_reward = boss.get_reward(next(filter(boss.filter_boss, _enemies))) if _enemies else 0

                        # distance奖励与knight到达enemy的距离相关
                        # 取值范围(-1, 2)
                        # x_reward, y_reward = self._get_distance_reward(_knight, _enemies) if _enemies else (0, 0)

                        # wall奖励与knight是否撞墙相关
                        # 取值范围(-2, 0)
                        # wall_reward = game.get_wall_reward(_knight)

                        # action_reward = boss_reward + y_reward
                        # move_reward = knight_reward + x_reward + wall_reward

                        action_reward = boss_reward
                        move_reward = knight_reward

                        game.update(_knight)
                        # draw_ui(collider['Knight'], collider['Enemies'], collider['Attacks'])

                        knight_points = _knight['position']
                        enemy_points = list(chain(*[enemy['position'] for enemy in _enemies]))

                        state, move_index, action_index, move_random, action_random = \
                            self._fight_boss(knight_points, enemy_points)
                        pool.record(state, move_index, action_index, move_reward, action_reward)

                        print(receive_time, 'Turing',
                              '[random]:' if move_random else '[prediction]:',
                              'move', game.move_names[move_index], 'get', move_reward,
                              '[random]:' if action_random else '[prediction]:',
                              'action', game.action_names[action_index], 'get', action_reward,
                              )
                    case 'GG_Workshop':
                        self._end_boss(receive_time)
        return scene

    def _before_boss(self, boss_start: datetime):
        """ 进入BOSS场景, 需要初始化一些操作
        """
        self.boss_start = boss_start
        # self.delay_time = []
        knight.reset()
        boss.reset()
        start_draw()

    def _get_distance_reward(self, _knight: dict, enemies_data: list) -> (float, float):
        _boss = next(filter(boss.filter_boss, enemies_data))

        (boss_left, boss_top), (boss_right, boss_bottom) = tuple(tuple(point) for point in _boss['position'])
        (knight_left, knight_top), (knight_right, knight_bottom) = tuple(tuple(point) for point in _knight['position'])

        x_distance = self._get_distance(knight_left, knight_right, boss_left, boss_right)
        y_distance = self._get_distance(knight_top, knight_bottom, boss_top, boss_bottom)

        attack_x, attack_up, attack_down = knight.attack_distance

        x_reward = knight.x_reward / attack_x * x_distance if x_distance <= attack_x else \
            self._get_linear_reward(x_distance, attack_x, knight.x_reward, knight.x_distance, 0)

        y_reward = 0 if x_distance > 2 * attack_x else \
            knight.y_reward / attack_down * y_distance if y_distance <= attack_down else \
                self._get_linear_reward(y_distance, attack_down, knight.y_reward, knight.y_distance, 0)

        return x_reward, y_reward

    def _get_linear_reward(self, x: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        使用另外一个线性函数 (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1)
        # y = (y2 - y1)*(x - x1)/(x2 - x1) + y1
        """
        return (y2 - y1) * (x - x1) / (x2 - x1) + y1

    def _get_distance(self, a: float, b: float, c: float, d: float) -> float:
        return 0 if c < a < d or c < b < d else min(min(abs(a - c), abs(a - d)), min(abs(b - c), abs(b - d)))

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
        game.step(game.moves[move_index], game.actions[action_index])

        return state_fixed, move_index, action_index, move_random, action_random

    def _end_boss(self, end_time: datetime):
        """ 离开BOSS场景, 需要统计一些操作
        """
        # 本场统计
        print("Fight boss within:", (end_time - self.boss_start).seconds, "s")
        # print("Take actions:", len(self.delay_time))
        # if len(self.delay_time) > 0:
        #     average_time_delay = np.mean(list(map(lambda delay: delay.total_seconds(), self.delay_time)))
        #     print("Average time delay:", average_time_delay * 1000, "ms")
        # 记录下boss的血量
        tf.summary.scalar('BOSS HP', boss.hp, self.fight_count)
        # 统计每个动作的使用次数
        with tf.name_scope(f"fight_count[{self.fight_count}]"):
            move_frequency = [0] * game.len_moves
            action_frequency = [0] * game.len_actions
            # 随机操作时, 虽然记录的是预测的价值, 但是并没有采用
            for index in [np.array(move_value).argmax() for move_value in knight.move_values]:
                move_frequency[index] += 1
            for index in [np.array(action_value).argmax() for action_value in knight.action_values]:
                action_frequency[index] += 1
            move_frequency = json.dumps(dict(zip(game.move_names, move_frequency)))
            action_frequency = json.dumps(dict(zip(game.action_names, action_frequency)))
            print(f'move frequency: {move_frequency}')
            print(f'action frequency: {action_frequency}')
            tf.summary.text('move action frequency', json.dumps((move_frequency, action_frequency)),
                            len(knight.action_values) + 1)
        writer.flush()
        self.fight_count += 1

        # 准备下一场挑战
        game.challenge()
        end_draw()


def get_save_path(prefix, time_format='%m-%d_%H:%M:%S'):
    save_file_path = prefix + start_time.strftime(time_format)
    if not os.path.isdir(save_file_path):
        os.makedirs(save_file_path)
    return save_file_path


if __name__ == '__main__':
    pyautogui.FAILSAFE = False
    start_time = datetime.now()
    writer = tf.summary.create_file_writer(get_save_path('log/'))
    game = Game()
    knight = Knight()
    boss = Boss()
    pool = Pool('save/' + sys.argv[1])
    agent = Agent('save/' + sys.argv[1])
    Turing().start()
