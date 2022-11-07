# coding: utf-8

import math
import os
import socket
import struct
import sys


def slice_pi(min_k: int, max_k: int) -> float:
    s = 0.0
    for k in range(min_k, max_k):
        s += 1.0 / (2 * k + 1) / (2 * k + 1)
    return s


# 使用无穷级数计算π:
# https://www.zhihu.com/question/402311979
# 这里用的是Fouier(傅里叶)级数
# (π^2) / 8 = 1 + 1/3^2 + 1/5^2 + 1/7^2......
# 程序把加法计算分配到了10个进程中去执行, 最后再合并计算结果
def pi(n):
    server_address = "/tmp/pi_sock"  # 套接字对应的文件名
    # os.unlink(server_address)  # 移除套接字文件
    servsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    servsock.bind(server_address)
    servsock.listen(10)  # 监听子进程连接请求
    child_process_ids = []
    unit = n // 10
    for i in range(10):  # 分10个子进程
        min_k = unit * i
        max_k = min_k + unit
        # 在Unix/Linux下，可以使用fork()调用实现多进程。
        # 要实现跨平台的多进程，可以使用multiprocessing模块。
        # fork()调用一次，返回两次，因为操作系统自动把当前进程（称为父进程）复制了一份（称为子进程），然后，分别在父进程和子进程内返回。
        # 子进程永远返回0，而父进程返回子进程的ID。
        pid = os.fork()
        if pid > 0:
            # 父进程, 保存子进程的id,
            child_process_ids.append(pid)
        else:
            # 子进程, 计算部分无穷级数
            servsock.close()  # 子进程要关闭servsock引用
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(server_address)  # 连接父进程套接字

            s = slice_pi(min_k, max_k)  # 子进程开始计算
            sock.sendall(struct.pack('f', s))  # send方法发送的数据比较少, sendall只能发送bytes
            sock.close()  # 关闭连接
            sys.exit(0)  # 子进程结束
    sums = []
    for pid in child_process_ids:
        # accept是等待函数
        conn, info = servsock.accept()  # 接收子进程连接, 因为是用的文件, 所以没有端口地址
        print("info: ", info)
        sums.append(struct.unpack('<f', conn.recv(1024))[0])  # recv收到的是bytes, 需要什么类型需要自己转换
        conn.close()  # 关闭连接
    for pid in child_process_ids:
        os.waitpid(pid, 0)  # 等待子进程结束

    servsock.close()  # 关闭套接字
    os.unlink(server_address)  # 移除套接字文件
    return math.sqrt(sum(sums) * 8)


# 多进程学习: https://www.liaoxuefeng.com/wiki/1016959663602400/1017628290184064
# 代码: https://www.cnblogs.com/-wenli/p/13100104.html
print(pi(1000000))
