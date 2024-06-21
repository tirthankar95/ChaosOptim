import torch
import torch.nn as nn 
import torch.optim as optim
import multiprocessing as mpc 
from GLOBAL import global_module as g 
import copy
from typing import List 

# def load_parameters(model, fName):
#     with torch.no_grad():
#         for idx, layer in enumerate(model.parameters()):
#             layer = torch.load(f'dat/{fName}_layer{idx}')

# def save_parameters(model, fName):
#     with torch.no_grad():
#         for idx, layer in enumerate(model.parameters()):
#             torch.save(layer, f'dat/{fName}_layer{idx}')

def trajectory_unroll(model):
    
'''
'''
def model_iter() -> List[int]:
    pass

'''
Test:
for param1, param2 in zip(obj1.parameters(), obj.parameters()):
    assert torch.equal(param1, param2)
'''
def init_wt(cModel, pModel):
    cModel.load_state_dict(pModel.state_dict())
    for idx, lc in enumerate(cModel.parameters()):
        lc.data = lc.data + torch.normal(0, g.std, size = lc.shape)

def run_nn(nn_class, ip_size):
    join = []
    parent_model = nn_class(ip_size)
    child_models = []
    # COPYING SIMILAR WTS TO CHILD FROM PARENT.
    for _ in range(g.child_n):
        child_models.append(nn_class(ip_size))
        p = mpc.Process(target = init_wt, args = (child_models[-1], parent_model))
        join.append(p)
        p.start()
    for p in join:
        p.join()
    # TRAJECTORY UNROLL. 
    join = []
    for _ in range(g.child_n):
        p = mpc.Process(target = model_iter, args = )
        join.append(p)
        p.start()
    for p in join:
        p.join()
    

if __name__ == '__main__':
    from NNModel import model1 as m1 
    run_nn(m1.NNetwork, ip_size = 10)