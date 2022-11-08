# coding: utf-8
import os
import socket

if __name__ == '__main__':
    # 配置socket
    socket_service = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定socket文件
    socket_service.bind(("127.0.0.1", 9203))
    # 如果队列满了, 请求会被拒绝, 这里多创建些, 防止队列满了
    socket_service.listen(10)
    # 使用一个死循环, 不断的接受客户端的消息
    while True:
        # 接收其他进程连接, 因为是用的文件, 所以没有端口地址
        connection, address = socket_service.accept()
        # 收到的原始数据
        origin_data = connection.recv(1024)
        print(origin_data)
        # 解析为string类型
        string_data = origin_data.decode('utf-8')
        print(string_data)
