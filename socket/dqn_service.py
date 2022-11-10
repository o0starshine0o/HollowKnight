# coding: utf-8
import random
import socket

import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.activations import softmax


class DQN:

    def __init__(self, input_size: int, batch_size: int = 1, learning_rate: float = 0.001):
        self.actions = [None, 'a', 'd', 'k']
        self.model = self.__build_deep_q_network__(Input((input_size,), batch_size), learning_rate)

    def __build_deep_q_network__(self, input_layer: Input, learning_rate: float = 0.001):
        x = Dense(128)(input_layer)
        x = Flatten()(x)
        output_layer = Dense(len(self.actions), softmax)(x)
        model = Model(input_layer, output_layer)
        model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())
        return model

    def get_action(self, state: np.ndarray):
        # 根据不同的batch_size, 这里会返回多个选取的动作
        # 因为输入的batch_size是1, 这里输出的shape就是(1, 4)
        prediction = self.__get_prediction__(state)
        # print("get_action: ", prediction)
        action_index = prediction[0].numpy().argmax()
        # return self.actions[action_index]
        # 先返回一个随机的动作, 之后再补上DQN网络的动作
        return random.choice(self.actions)

    @tf.function
    def __get_prediction__(self, state: np.ndarray):
        return self.model(state)


class Game:

    def __init__(self):
        pass

    def step(self, action: str):
        if action is None:
            return
        pyautogui.press(action)


class Turing:

    def __init__(self, game: Game, dqn: DQN):
        self.game = game
        self.dqn = dqn
        self.boss = False
        # 配置socket
        self.socket_service = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 绑定socket文件
        self.socket_service.bind(("127.0.0.1", 9203))
        # 如果队列满了, 请求会被拒绝, 这里多创建些, 防止队列满了
        self.socket_service.listen(10)
        # 不解释
        print("Make AI Great Again")

    def start(self):
        # 使用一个死循环, 不断的接受客户端的消息
        while True:
            # 接收其他进程连接, 因为是用的文件, 所以没有端口地址
            connection, address = self.socket_service.accept()

            while True:
                # 收到的原始数据
                origin_data = connection.recv(1024)
                # 如果没有数据, 就断开连接
                if not origin_data:
                    break
                # 解析为string类型
                string_data = origin_data.decode('utf-8')
                # 如果场景退出, 就结束本次AI过程
                if string_data == 'GG_Workshop':
                    self.boss = False
                # 如果进入了BOSS场景, 就开始AI操作
                if string_data == 'GG_Hornet_2':
                    self.boss = True
                # 如果在BOSS战, 就开始AI操作
                if self.boss:
                    # 把得到的状态给DQN, 拿到action
                    action = self.dqn.get_action(np.full(shape=(1, 32), fill_value=random.random(), dtype='float32'))
                    # print("Turing take action: ", action, "for state: ")
                    # 把dqn计算得到的action给游戏
                    self.game.step(action)


if __name__ == '__main__':
    Turing(Game(), DQN(32, 1)).start()
