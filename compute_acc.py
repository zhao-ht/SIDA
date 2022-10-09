import os
import torch

import re
import numpy as np


import sys
result=[]
with open('./experiments/ckpt/visda17_test_22/preds.txt','r') as f:
	for line in f:
		result.append(re.split('./|/|_| ', line.strip('\n')))

cats={}
i=0
with open('./experiments/dataset/VisDA-2017/category.txt','r') as f:
    for line in f:
        cats[line.strip('\n')]=i
        i=i+1


tag=[]
pred=[]
for i in result:
    tag.append(cats[i[6]])
    pred.append(int(i[8]))

tag=torch.tensor(tag)
pred=torch.tensor(pred)

cr=[]
for i in range(12):
    cr.append(100*float(((tag==i)*(pred==i)).sum()/(tag==i).sum()))

print(cr)