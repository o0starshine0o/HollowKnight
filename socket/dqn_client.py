import socket
import typing

_data = [
    "{\"scene\":null,\"time\":\"11/13/2022 4:33:43 PM.210\",\"knight_points\":null,\"enemy_points\":[]}",
    "{\"scene\":\"GG_Workshop\",\"time\":\"11/13/2022 4:34:58 PM.646\",\"knight_points\":[-1940.06946,504.584229,"
    "-1973.4093,419.150757],\"enemy_points\":[]}",
    "{\"scene\":\"GG_Hornet_2\",\"time\":\"11/13/2022 4:34:58 PM.646\",\"knight_points\":[-1940.06946,504.584229,"
    "-1973.4093,419.150757],\"enemy_points\":[]}",
    "{\"scene\":\"GG_Hornet_2\",\"time\":\"11/13/2022 4:35:51 PM.995\",\"knight_points\":[1404.144,"
    "-866.055664,1437.48389,-951.489136],\"enemy_points\":[[1339.94312,-780.64386,1280.28491,-951.6559],"
    "[1461.41333,-893.2883,1340.41077,-908.9575],[1461.41333,-859.281738,1340.41077,-874.9507],[1609.34192,"
    "-650.6317,1385.41516,-950.6137],[1463.27014,-622.9189,1069.28552,-952.8096]]}",
    "{\"scene\":\"GG_Workshop\",\"time\":\"11/13/2022 4:34:58 PM.646\",\"knight_points\":[-1940.06946,504.584229,"
    "-1973.4093,419.150757],\"enemy_points\":[]}",
]


def _send(message: bytes):
    client.send(message)
    client.recv(1024)  # 接收客户端确认


def _send_repeat(count: int, messages: typing.Callable[[int], str]):
    for index in range(0, count):
        _send(str.encode(messages(index)))


def _get_message(index: int):
    return _data[index]


if __name__ == '__main__':

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9203))

    try:
        _send_repeat(len(_data), _get_message)
    except ConnectionResetError as e:
        print(e)

    client.close()

    exit()
