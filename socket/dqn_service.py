# coding: utf-8
import random
import socket

import pyautogui


class DQN:

    def __init__(self):
        self.actions = ['a', 'd', 'k', None]

    def get_action(self):
        # 先返回一个随机的动作, 之后再补上DQN网络的动作
        return random.choice(self.actions)


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
                print(string_data)
                # 如果场景退出, 就结束本次AI过程
                if string_data == 'GG_Workshop':
                    self.boss = False
                # 如果进入了BOSS场景, 就开始AI操作
                if string_data == 'GG_Hornet_2':
                    self.boss = True
                # 如果在BOSS战, 就开始AI操作
                if self.boss:
                    # 把得到的状态给DQN, 拿到action
                    action = self.dqn.get_action()
                    print("Turing take action: ", action, "for state: ")
                    # 把dqn计算得到的action给游戏
                    self.game.step(action)


if __name__ == '__main__':
    Turing(Game(), DQN()).start()
