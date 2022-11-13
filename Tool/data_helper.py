# coding: utf-8
import json


def parse_data(origin_data: bytes):
    try:
        # 解析为string类型
        string_data = origin_data.decode('utf-8')
        # 保存为字典类型
        return json.loads(string_data)
    except json.JSONDecodeError as error:
        print(error)
