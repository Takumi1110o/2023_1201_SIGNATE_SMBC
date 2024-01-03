import pandas as pd
import yaml
import os

# with open(R'params.yaml') as file:
#     yml = yaml.safe_load(file)

# # path = yml['path']
# train = pd.read_csv("..exp/data/train.csv")
# train.head()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
file_path = os.path.join(parent_dir, 'data')
print(file_path)