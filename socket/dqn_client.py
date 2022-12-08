import json
import socket
import sys
import time
from datetime import datetime


class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


class Knight:
    def __init__(self):
        self.position = [[403, 619], [370, 533]]
        self.hp = 9
        self.mp = 0
        self.state = 'airborne'
        self.velocity = [0, -0.948]
        self.invulnerable = False
        self.dashing = False
        self.superDashing = False
        self.jumping = False
        self.doubleJumping = False
        self.falling = True
        self.attacking = False
        self.canCast = True
        self.canSuperDash = False

    def update(self, index: int):
        pass


class Enemy:
    hp = 800
    maxHp = 800
    name = "Hornet Boss 2"
    isActive = True
    position = [[395, 350], [514, 350]]

    def __init__(self, index=0):
        super().__init__()
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
        self.position = [[395, 350], [514, 350]]

    def _init_attack(self):
        self.hp = 800
        self.maxHp = 800
        self.name = "Hit ADash"
        self.isActive = True
        self.position = [[437, 324], [443, 310], [345, 244]]

    def update(self, index: int):
        pass


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
        self.Enemies = [Enemy(i) for i in range(2)]
        self.Attacks = [Attack(i) for i in range(1)]

    def update(self, index: int):
        self.Knight.update(index)
        for enemy in self.Enemies:
            enemy.update(index)
        for attack in self.Attacks:
            attack.update(index)
        pass


class Data:

    def __init__(self, count):
        super().__init__()
        self.scene = ""
        self.time = ''
        self.count = count
        self.collider = Collider()

    def _to_workshop(self):
        self.scene = 'GG_Workshop'

    def _to_hornet(self):
        self.scene = 'GG_Hornet_2'

    def update(self, index: int):
        match index:
            case 0:
                self._to_workshop()
            case 1:
                self._to_hornet()
            case self.count:
                self._to_workshop()
        self.collider.update(index)
        self.time = datetime.now().strftime('%m/%d/%Y %I:%M:%S %p.%f')[:-3]


def _send(message: bytes | str):
    match message:
        case str():
            message = str.encode(message)
    print(message)
    client.send(message)
    client.recv(1024)  # 接收客户端确认


if __name__ == '__main__':

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9203))

    length = sys.argv[1] if len(sys.argv) > 1 else ''
    length = 3 + int(length) if length.isdigit() else 10

    data = Data(length - 1)

    try:
        for sequence in range(length):
            data.update(sequence)
            _send(json.dumps(data, cls=JsonEncoder))
            time.sleep(0.1)
    except ConnectionResetError as e:
        print(e)

    client.close()

    exit()
