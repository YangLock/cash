import torch

data_dict = torch.load("../heprediction_data/splits/citeseersplit0.pt")

print(data_dict.keys())

print(len(data_dict['train_only_pos']))
print(len(data_dict['train_sns']))
