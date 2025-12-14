import numpy as np
import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, target_pc, output_pc):
        # both pcs are 1024 x 3
        sum_sq_a = np.sum(output_pc**2, axis=1, keepdims=True)
        sum_sq_b = np.sum(target_pc**2, axis=1, keepdims = True).T

        dot = 2* np.dot(output_pc, target_pc.T)

        #forward
        squared_dst = sum_sq_a - dot + sum_sq_b
        squared_dst[squared_dst < 0] = 0
        distances = np.sqrt(squared_dst)
        loss_a_to_b = np.sum(np.min(distances, axis=1))

        #backward
        sum_sq_a = sum_sq_a.T
        sum_sq_b = sum_sq_b.T
        dot = 2* np.dot(target_pc, output_pc.T)
        squared_dst = sum_sq_b - dot + sum_sq_a
        squared_dst[squared_dst < 0] = 0
        distances = np.sqrt(squared_dst)
        loss_b_to_a = np.sum(np.min(distances, axis=1))

        return (loss_a_to_b + loss_b_to_a) / 1000.0

if __name__ == "__main__":
    a = np.array([[0,2],
            [0,5],
            [0,7]])
    b = np.array([[1,2],
            [10,5],
            [2,7]])
    c = ChamferLoss()
    c.forward(a, b)

def forward(target_pc, output_pc):
    # both pcs are 1024 x 3
    sum_sq_a = np.sum(output_pc**2, axis=1, keepdims=True)
    sum_sq_b = np.sum(target_pc**2, axis=1, keepdims = True).T

    dot = 2* np.dot(output_pc, target_pc.T)

    #forward
    squared_dst = sum_sq_a - dot + sum_sq_b
    squared_dst[squared_dst < 0] = 0
    distances = np.sqrt(squared_dst)
    loss_a_to_b = np.sum(np.min(distances, axis=1))

    #backward
    sum_sq_a = sum_sq_a.T
    sum_sq_b = sum_sq_b.T
    dot = 2* np.dot(target_pc, output_pc.T)
    squared_dst = sum_sq_b - dot + sum_sq_a
    squared_dst[squared_dst < 0] = 0
    distances = np.sqrt(squared_dst)
    loss_b_to_a = np.sum(np.min(distances, axis=1))

    return (loss_a_to_b + loss_b_to_a) / 1000.0