import json
import random
import socket
import sys
from datetime import datetime

from DQN_HollowKnight.Tool.data_helper import parse_data
from DQN_HollowKnight.Tool.ui_helper import start_draw, end_draw, draw_ui


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class Env:
    x = 293, 2260
    y = 173, 1080


class Knight:
    def __init__(self):
        self.position = [[393, 283], [436, 173]]
        self.hp = 9
        self.mp = 0
        self.state = 'airborne'
        self.velocity = [0, 0]
        self.invulnerable = False
        self.dashing = False
        self.superDashing = False
        self.jumping = False
        self.doubleJumping = False
        self.falling = False
        self.attacking = False
        self.canCast = True
        self.canSuperDash = True
        self.touchingWall = False
        self.facingRight = True

    def update(self, index: int, move_index: int, action_index: int):
        match move_index:
            case 1:
                self.velocity = [-random.randint(40, 60), 0]
                self.facingRight = False
            case 2:
                self.velocity = [random.randint(40, 60), 0]
                self.facingRight = True

        # left
        self.position[0][0] += self.velocity[0]
        self.position[0][0] = max(min(self.position[0][0], Env.x[1]), Env.x[0])
        # top
        self.position[0][1] += self.velocity[1]
        # right
        self.position[1][0] = self.position[0][0] + 43
        # bottom
        self.position[1][1] = self.position[0][1] - 110

        self.touchingWall = self.position[0][0] == Env.x[1] or self.position[0][0] == Env.x[0]


class Enemy:
    hp = 800
    maxHp = 800
    name = "Hornet Boss 2"
    isActive = True
    position = [[1695, 283], [1814, 173]]
    move = 50

    def __init__(self, index=0):
        match index:
            case 0:
                self._init_hornet()
            case 1:
                self._init_attack()

    def _init_hornet(self):
        self.hp = 800
        self.maxHp = 800
        self.name = "Hornet Boss 2"
        self.isActive = True
        self.position = [[1695, 283], [1814, 173]]

    def _init_attack(self):
        self.hp = 800
        self.maxHp = 800
        self.name = "Hit ADash"
        self.isActive = True
        self.position = [[437, 324], [443, 310], [345, 244]]

    def update(self, index: int):
        if self.position[0][0] <= Env.x[0] or self.position[0][0] >= Env.x[1]:
            self.move = -1 * self.move
        self.position[0][0] += self.move
        self.position[1][0] = self.position[0][0] + 119


class Attack:

    def __init__(self, index=0):
        self.name = 'Slash'
        self.isActive = True
        self.position = [[346, 314], [250, 252], [231, 173], [286, 132], [525, 101], [537, 349]]

    def update(self, index: int):
        pass


class Collider:

    def __init__(self):
        self.Knight = Knight()
        self.Enemies = [Enemy(i) for i in range(1)]
        self.Attacks = [Attack(i) for i in range(0)]

    def update(self, index: int, move_index: int, action_index: int):
        self.Knight.update(index, move_index, action_index)
        for enemy in self.Enemies:
            enemy.update(index)
        for attack in self.Attacks:
            attack.update(index)
        pass


class Data:

    def __init__(self, count):
        """
        Params
        count 需要执行的次数
        """
        self.scene = ""
        self.time = ''
        self.count = count
        self.collider = Collider()

    def _to_workshop(self):
        self.scene = 'GG_Workshop'

    def _to_hornet(self):
        self.scene = 'GG_Hornet_2'

    def update(self, index: int, move_index: int, action_index: int):
        """
        Params
        index 当前执行动作序列
        move_index 负责移动的index
        action_index 负责动作的index
        """
        match index:
            case 0:
                self._to_workshop()
            case 1:
                self._to_hornet()
            case self.count:
                self._to_workshop()
        self.collider.update(index, move_index, action_index)
        self.time = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p.%f')[:-3]


def _try_step(message: bytes) -> bytes:
    try:
        client.send(message)
        return client.recv(2048)
    except ConnectionResetError as e:
        print(e)
    except BrokenPipeError as e:
        print(e)


def _step(message: str):
    print(f'step: {message}')
    origin_data = _try_step(str.encode(message))
    return tuple(parse_data(origin_data)) if origin_data else (0, 0)


def _send_data():
    move_index, action_index = 0, 0

    start_draw()
    try:
        for index in range(data.count):
            data.update(index, move_index, action_index)
            json_data = json.dumps(data, cls=JsonEncoder)
            dict_data = parse_data(json_data)
            collider = dict_data['collider']
            draw_ui(collider['Knight'], collider['Enemies'], collider['Attacks'])
            move_index, action_index = _step(json_data)
    except KeyboardInterrupt as e:
        print(e)
    finally:
        end_draw()


def _connect():
    try:
        client.connect(("127.0.0.1", 9203))
        _send_data()
        client.close()
    except ConnectionRefusedError as e:
        print(e)


def _get_sequence_length() -> int:
    length = sys.argv[1] if len(sys.argv) > 1 else ''
    length = 3 + int(length) if length.isdigit() else 10
    return length


if __name__ == '__main__':
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data = Data(_get_sequence_length() - 1)
    _connect()
    exit()
