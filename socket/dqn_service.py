# coding: utf-8
import _thread
import os
import random
import socket
import sys
import time
from datetime import datetime

import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from DQN_HollowKnight.Tool.data_helper import parse_data


class Knight:
    def __init__(self):
        self.hp = 9

    def reset(self):
        self.hp = 9


class Boss:
    def __init__(self, name='Hornet Boss 2'):
        self.hp = 800
        self.name = name

    def reset(self, name='Hornet Boss 2'):
        self.hp = 800
        self.name = name


class Pool:

    def __init__(self, save_path: str = None, size: int = 10 * 60 * 60):
        self.size = size
        if self._file_exist(save_path):
            self.states = np.load(save_path + '/states.npy')
            self.actions = np.load(save_path + '/actions.npy')
            self.rewards = np.load(save_path + '/rewards.npy')
            self.current, self.count, size = tuple(np.load(save_path + '/meta.npy'))
            print(f'load pool:[{self.current}|{self.count}|{size}]', save_path)
        else:
            self.states = np.empty((self.size, 32))
            self.actions = np.empty(self.size, dtype=int)
            self.rewards = np.zeros(self.size)
            self.current = 0
            self.count = 0

    def record(self, state: np.ndarray, action: int, reward: float):
        self.states[self.current] = state
        self.actions[self.current] = action
        # reward 为上一步的action的奖励, 所以往上推一步
        self.rewards[self.current - 1] = reward
        self.current += 1
        self.count = max(self.count, self.current)
        self.current %= self.size

    def recall(self, size: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Returns:
            states: 选取的状态S0
            actions: 此状态下选取的动作A0
            rewards: 选取动作的奖励R0
            next_states: 到达的下一个状态S1
        """
        if self.count < size:
            print('size < count', 'not enough data')
            return None, None, None, None
        indices = np.array(random.sample(range(self.count), size))
        next_indices = (indices + 1) % self.count
        return self.states[indices], self.actions[indices], self.rewards[indices], self.states[next_indices]

    def save(self, save_path: str):
        if save_path and os.path.exists(save_path):
            np.save(save_path + '/states.npy', self.states)
            np.save(save_path + '/actions.npy', self.actions)
            np.save(save_path + '/rewards.npy', self.rewards)
            np.save(save_path + '/meta.npy', np.array([self.current, self.count, self.size]))
            print(f'save pool: {save_path} [{self.current}|{self.count}|{self.size}]')

    def _file_exist(self, save_path: str = None) -> bool:
        return save_path \
               and os.path.exists(save_path + '/states.npy') \
               and os.path.exists(save_path + '/actions.npy') \
               and os.path.exists(save_path + '/rewards.npy') \
               and os.path.exists(save_path + '/meta.npy')


class Agent:

    def __init__(self, save_path: str = None, input_size=32, gamma=0.99, learning_rate=0.01, batch_size=32):
        self.actions = [None, 'a', 'd', 'j', 'k']
        if save_path and os.path.exists(save_path):
            self.model = tf.keras.models.load_model(save_path + '/model.tf')
            print(f'load model:', save_path + '/model.tf')
        else:
            self.model = self.__build_deep_q_network__(Input(input_size), learning_rate)
        # 程序启动就开始学习, 生命不停, 学习不止
        self.learn_thread = _thread.start_new_thread(self._learn, (gamma, batch_size))

    def __build_deep_q_network__(self, input_layer: Input, learning_rate: float = 0.01):
        hidden = Dense(128)(input_layer)
        hidden = Dense(256)(hidden)
        hidden = Dense(512)(hidden)
        hidden = Dense(1024)(hidden)
        hidden = Dense(2048)(hidden)
        hidden = Dense(256)(hidden)
        hidden = Dense(32)(hidden)
        flatten = Flatten()(hidden)
        output_layer = Dense(len(self.actions), softmax)(flatten)
        model = Model(input_layer, output_layer)
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
        return model

    def get_action(self, state: np.ndarray, is_random=False):
        """
        返回一个action的索引
        """
        # 经验池比较小, 采取随机操作, 增大经验池
        if is_random:
            return random.randint(0, len(self.actions) - 1), is_random
        # 根据不同的batch_size, 这里会返回多个选取的动作
        # 因为输入的batch_size是1, 这里输出的shape就是(1, 4)
        prediction = self.__get_prediction__(state)
        return prediction[0].numpy().argmax(), is_random

    def _learn(self, gamma=0.99, batch_size=32, empty_sleep_time=1):
        step = 0
        start = datetime.now()
        writer = tf.summary.create_file_writer('log')
        try:
            with writer.as_default():
                while True:
                    states, actions, rewards, next_states = pool.recall(batch_size)
                    if states is None or actions is None or rewards is None or next_states is None:
                        # 避免无数据时空转
                        time.sleep(empty_sleep_time)
                        continue

                    # 核心梯度下降过程
                    loss = self._learn_kernel(states, actions, rewards, next_states, gamma)

                    step += 1

                    # 保存损失信息
                    if step % 100 == 0:
                        print(f'step: {step}, loss {loss.numpy()}')
                        tf.summary.scalar('loss', loss.numpy(), step)
                        writer.flush()

                    # 输出一些信息表明线程正常运行
                    if step % 1000 == 0:
                        # 不解释
                        print(f'good good study, day day up {datetime.now() - start}')
                        start = datetime.now()
        except KeyboardInterrupt:
            writer.close()

    def _learn_kernel(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray,
                      gamma=0.99) -> tf.Tensor:
        # 1, 根据当前网络, 计算next_states下所有action的value, 并且选取价值最大的那个动作的值, 作为next_q_max
        # next_actions_value_max = max(Q(S_1, W)), shape: (32, )
        next_actions_value_max = self.__get_prediction__(next_states).numpy().max(axis=1)
        # 2, 添加gamma, reward作为target_q
        # target_q = R_0 + gamma * next_q_max, shape: (32, )
        target_q = rewards + gamma * next_actions_value_max
        with tf.GradientTape() as tape:
            # 3, 根据当前网络, 重新计算states下所有actions的value
            # action_values = Q(S0, W), shape: (32, 5)
            action_values = self.__get_prediction__(states)
            # 当时选中的action, 转换为one_hot, shape: (32, 5)
            one_hot_actions = tf.keras.utils.to_categorical(actions, len(self.actions))
            # 根据当时选择的action(可能是随机选择的, 不一定是argmax), 和现在的网络参数, 计算q值, shape: (32, )
            q = tf.reduce_sum(tf.multiply(action_values, one_hot_actions), axis=1)
            # 4, 使用2者的MSE作为loss
            # loss = (target_q - q)^2
            loss = self.model.loss(target_q, q)
            # 5, 计算梯度
            model_gradients = tape.gradient(loss, self.model.trainable_variables)
        # 6, 反向传播, 更新model的参数
        self.model.optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))

        return loss

    def save(self, save_path: str = None):
        if save_path and os.path.exists(save_path):
            self.model.save(save_path + '/model.tf')
            print('save model:', save_path + '/model.tf')

    @tf.function
    def __get_prediction__(self, state: np.ndarray):
        return self.model(state)


class Game:
    """
    负责与游戏交互
    """

    def __init__(self):
        pass

    def step(self, action: str):
        if action is None:
            return
        pyautogui.press(action)

    def challenge(self):
        _thread.start_new_thread(self._challenge, ())
        pass

    def _challenge(self):
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
        # 记录下HollowKnight发送状态到收到动作的时间延迟
        self.delay_time = []
        # 不解释
        print("Make AI Great Again")

    def start(self):
        # 使用一个死循环, 不断的接受客户端的消息
        try:
            while True:
                self._game_start()
        except KeyboardInterrupt:
            print('end')

    def save(self, time_format='%m-%d_%H:%M:%S'):
        # Create the folder for saving the agent
        save_file_path = 'save/' + datetime.now().strftime(time_format)
        if not os.path.isdir(save_file_path):
            os.makedirs(save_file_path)
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
                origin_data = connection.recv(1024)
                # 如果没有数据, 就断开连接
                if not origin_data:
                    break
                # 处理当前帧
                scene = self._frame_current(origin_data, scene)
                # 反馈给客户端
                connection.send(str.encode(datetime.now().strftime(self._time_format)[:-3]))
            except ConnectionResetError as exception:
                print(exception)

        # 保存当前记录
        self.save()

    def _frame_current(self, origin_data: bytes, scene: str):
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
        match scene:
            case 'GG_Workshop':
                match json_data['scene']:
                    case 'GG_Hornet_2':
                        self._before_boss(receive_time)
            case 'GG_Hornet_2':
                match json_data['scene']:
                    case 'GG_Hornet_2':
                        pre_reward = self._get_reward(json_data['hp'], json_data['enemies'])
                        knight_points, enemy_points = json_data['knight_points'], json_data['enemy_points']
                        state, action_index, is_random = self._fight_boss(knight_points, enemy_points)
                        # todo: delete, 验证模型可学习
                        if agent.actions[action_index] == 'k':
                            pre_reward = 1
                        pool.record(state, action_index, pre_reward)
                        if pre_reward:
                            print(receive_time, "Turing get", pre_reward,
                                  "with random:" if is_random else "with action:", agent.actions[action_index],
                                  "for state:")
                    case 'GG_Workshop':
                        self._end_boss(receive_time)
        return json_data['scene']

    def _before_boss(self, start_time: datetime):
        """ 进入BOSS场景, 需要初始化一些操作
        """
        self.boss_start = start_time
        self.delay_time = []
        knight.reset()
        boss.reset()

    def _get_reward(self, hp: int, enemies: list):
        """ 先获取到奖励
        """
        # knight 生命计算
        reward = -10 * (knight.hp - hp)
        knight.hp = hp
        # boss 生命计算
        if len(enemies) > 0:
            enemy = next(enemy for enemy in enemies if enemy['name'] == boss.name)
            if enemy:
                reward += (boss.hp - enemy['hp'])
                boss.hp = enemy['hp']
        # 综合奖励
        return reward

    def _fight_boss(self, knight_points: [float], positions: list[list[float]]):
        """ knight的坐标, 长度为4, [left, top, right, bottom]
        enemies的坐标, 目前有5个, 先固定分配位置, 第二维长度为4, [left, top, right, bottom]
        """
        # 把list转换为ndarray
        positions.insert(0, knight_points)
        # 调整形状: (n,)
        state_flatten = np.array(positions).flatten()
        # (32,)
        state_fixed = np.pad(state_flatten, (0, 32 - len(state_flatten)))
        # (1, 32)
        state_reshape = np.reshape(state_fixed, (1, 32))
        # 把得到的状态给DQN, 拿到action
        action_index, is_random = agent.get_action(state_reshape, pool.current < pool.size / 2)
        # 把dqn计算得到的action给游戏
        game.step(agent.actions[action_index])

        return state_fixed, action_index, is_random

    def _end_boss(self, end_time: datetime):
        """ 离开BOSS场景, 需要统计一些操作
        """
        # 本场统计
        print("Fight boss within:", (end_time - self.boss_start).seconds, "s")
        print("Take actions:", len(self.delay_time))
        if len(self.delay_time) > 0:
            average_time_delay = np.mean(list(map(lambda delay: delay.total_seconds(), self.delay_time)))
            print("Average time delay:", average_time_delay * 1000, "ms")
        # 准备下一场挑战
        game.challenge()


if __name__ == '__main__':
    game = Game()
    knight = Knight()
    boss = Boss()
    pool = Pool('save/' + sys.argv[1])
    agent = Agent('save/' + sys.argv[1])
    Turing().start()
