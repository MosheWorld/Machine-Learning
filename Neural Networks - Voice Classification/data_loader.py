import os
from gcommand_loader import GCommandLoader
import torch

dataset = GCommandLoader('./gcommands/train')

test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=None,
        num_workers=os.cpu_count(), pin_memory=True, sampler=None)

for k, (input,label) in enumerate(test_loader):
    print(input.size(), print(label))
print(k)
