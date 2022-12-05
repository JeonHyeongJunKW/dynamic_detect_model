import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR_Loss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin= margin

    def forward(self,origin,target,label):
        euclidean_distance = F.pairwise_distance(origin, target, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class obj_Loss(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image1, image2):

        return 0

class EarlyStopping():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.last_loss =10000000

    def test(self, current_loss):
        if (current_loss - self.last_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
                return True
        else:
            self.last_loss =current_loss
            self.counter =0
        return False