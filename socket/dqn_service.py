# coding: utf-8
import random
import socket
from datetime import datetime

import numpy as np
import pyautogui
import tensorflow as tf
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.layers import Dense, Flatten, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v2.adam import Adam

from DQN_HollowKnight.Tool.data_helper import parse_data


class DQN:

    def __init__(self, input_size: int, batch_size: int = 1, learning_rate: float = 0.001):
        self.actions = [None, 'a', 'd', 'j', 'k']
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
    _time_format = '%m/%d/%Y %I:%M:%S %p.%f'

    def __init__(self, game: Game, dqn: DQN):
        self.game = game
        self.dqn = dqn
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
        while True:
            self._game_start()

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
                scene = self._frame_start(origin_data, scene)
                # 反馈给客户端
                connection.send(str.encode(datetime.now().strftime(self._time_format)[:-3]))
            except ConnectionResetError as exception:
                print(exception)

    def _frame_start(self, origin_data: bytes, scene: str):
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
                        action = self._fight_boss(json_data['knight_points'], json_data['enemy_points'])
                        print(receive_time, "Turing take action:", action, "for state:")
                    case 'GG_Workshop':
                        self._end_boss(receive_time)
        return json_data['scene']

    def _before_boss(self, time: datetime):
        """ 进入BOSS场景, 需要初始化一些操作
        """
        self.boss_start = time
        self.delay_time = []

    def _fight_boss(self, knight: list[float], enemies: list[list[float]]):
        # 把得到的状态给DQN, 拿到action
        action = self.dqn.get_action(np.full(shape=(1, 32), fill_value=random.random()))
        # 把dqn计算得到的action给游戏
        self.game.step(action)

        return action

    def _end_boss(self, end_time: datetime):
        """ 离开BOSS场景, 需要统计一些操作
        """
        # 本场统计
        print("Fight boss within:", (end_time - self.boss_start).seconds, "s")
        print("Take actions:", len(self.delay_time))
        if len(self.delay_time) > 0:
            average_time_delay = np.mean(list(map(lambda delay: delay.total_seconds(), self.delay_time)))
            print("Average time delay:", average_time_delay * 1000, "ms")


if __name__ == '__main__':
    Turing(Game(), DQN(32, 1)).start()
