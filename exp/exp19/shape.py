import pandas as pd
from datetime import datetime
import geocoder

class ChangeTheShape():
    def __init__(self, data):
        self.result_list = {}
        name_list = data['problems'].unique()
        for strings in name_list:
            count = sum(1 for char in strings if char.isupper())
            self.result_list[strings] = count

    def change_created_at(self, str_datetime):
        date = datetime.strptime(str_datetime, '%Y-%m-%d')
        day = date - datetime(year=2015, month=5, day=19)
        return int(day.days)

    def change_problems(self, x):
        problems_list = ['Stones', 'BranchOther', 'BranchLights',
                        'RootOther', 'TrunkOther', 'TrunkLights',
                        'MetalGrates', 'WiresRope', 'Sneakers']
        sum = 0
        for problem in problems_list:
            if problem in x:
                sum += 1
        return sum