import json


class Config:
    data = {}

    def load(self):
        config_file = open('../config.json')
        self.data = json.load(config_file)
        config_file.close()
        print('配置载入成功')
