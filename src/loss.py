import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class SimCLR_Loss(nn.Module):
    def __init__(self, margin,type="mean",distance="euclidean"):
        super().__init__()
        self.margin= margin
        self.type =type
        self.distance=distance
    def forward(self,origin,target,label):
        eps=1e-6
        descriptor_distance= None
        if self.distance =="euclidean":
            descriptor_distance = F.pairwise_distance(origin, target, keepdim = True)+eps#batch
        elif self.distance=="cosine" :
            descriptor_distance = F.cosine_similarity(origin, target).unsqueeze(1)#batch
        label =label.unsqueeze(1)
        loss_contrastive = (1-label) * torch.pow(descriptor_distance, 2) +\
                            (label) * torch.pow(torch.clamp(self.margin - descriptor_distance, min=0.0), 2)
        if type=="mean":
            return torch.mean(loss_contrastive)
        else :
            return torch.sum(loss_contrastive)


        

class obj_Loss(nn.Module):
    def __init__(self, type="kernel_based"):
        super().__init__()

    def calcuate_imageloss(self, gray, prob):
        height = prob.size(2)
        width = prob.size(3)
        origin_rate = int(gray.size(2)/height)#비율
        total_loss =0
        gray_rate = F.interpolate(gray,size=(int(gray.size(2)/origin_rate),int(gray.size(3)/origin_rate)),mode='bicubic',align_corners=False)
        #이미지 엣지가 아닌곳에서 확률간에 편차가 낮고,
        #이미지 엣지인 곳에서는 확률간에 편차가 높다.
        gray_rate, _ = torch.max(gray_rate,1,keepdim=True)
        weight_list = [[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]]
        weight_edge = torch.nn.Parameter(torch.FloatTensor(weight_list)).cuda()
        edge_image = F.conv2d(gray_rate,weight=weight_edge,padding=1)
        # print("edge이미지 형태",edge_image.shape)

        weight_list = [[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]]
        weight_mean = torch.nn.Parameter(torch.FloatTensor(weight_list)).cuda()
        # torch.mean()
        mean_prob = F.conv2d(prob,weight=weight_mean,padding=1)
        # print(edge_image.shape)
        # print(mean_prob.shape)
        return torch.mean(-torch.abs(mean_prob)*torch.abs(edge_image))

    def forward(self, origin1, image1, origin2, image2):
        loss_1 = self.calcuate_imageloss(origin1,image1)
        loss_2 = self.calcuate_imageloss(origin2,image2)
        return loss_1+loss_2

class EarlyStopping():
    def __init__(self, tolerance=3, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.last_loss =0
        self.start = False
        self.not_luck = 0
    def test(self, current_loss):
        if not self.start:
            if self.not_luck<3:
                self.not_luck+=1
                return False
            self.start = True
            self.last_loss =current_loss
            self.counter =0
            return False
        if (current_loss - self.last_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
                return True
        else:
            self.last_loss =current_loss
            self.counter =0
        return False