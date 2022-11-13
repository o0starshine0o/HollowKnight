import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("127.0.0.1", 9203))

client.send(b"GG_Hornet_2")
client.recv(1024)  # 接收客户端确认
for _ in range(0, 2):
    try:
        client.send(b"11/13/2022 11:11:30 AM.156")
        ack = client.recv(1024)  # 接收客户端确认
    except ConnectionResetError as e:
        print(e)
        break
client.send(b"GG_Workshop")
client.recv(1024)  # 接收客户端确认

client.close()
