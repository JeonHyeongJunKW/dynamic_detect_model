import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d

class LDOD_model(nn.Module):#Longterm-dynamic object detection
    def __init__(self,image_width, image_height,gamma=16,dilation_rate=4):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.gamma = gamma
        self.dilation_rate =dilation_rate
        self.s_channel = int(64/self.gamma)
        ## origin model
        self.base_module_1 = nn.Sequential(Conv2d(1,64,7,padding=3),
                                            Conv2d(64,64,5,padding=2))#채널수만 바뀌면됨
        #주 목적은 장면내에서 이미지 픽셀 크기여야한다.
        self.attention_block_spatial = nn.Sequential(Conv2d(64,self.s_channel,1,padding=0),
                                                    Conv2d(self.s_channel,self.s_channel,3,dilation=self.dilation_rate,padding=4),
                                                    Conv2d(self.s_channel,self.s_channel,3,dilation=self.dilation_rate,padding=4),
                                                    Conv2d(self.s_channel,1,1,padding=0),
                                                    nn.Sigmoid())#마지막에 0과 1사이의 값을 가진다.
        self.shared_MLP = nn.Sequential(nn.Linear(64,self.s_channel),
                                        nn.Linear(self.s_channel,64),
                                        nn.ReLU())
        self.pooling_module_1 = nn.Sequential(Conv2d(64,512,padding=1,stride=2,dilation=1,kernel_size=4))#H/2, W/2

    def attention_block_channel(self, featuremap):
        #장면의 특성을 요약하는 역할 
        avg_result = F.adaptive_avg_pool2d(featuremap,(1,1))# output : [10, 64, 1, 1]
        max_result = F.adaptive_max_pool2d(featuremap,(1,1))

        #squeeze
        avg_result = avg_result.squeeze(3).squeeze(2)
        max_result = max_result.squeeze(3).squeeze(2)

        avg_result_MLP = self.shared_MLP(avg_result)
        max_result_MLP = self.shared_MLP(max_result)
        # print(avg_result_MLP.shape)
        # print(max_result_MLP.shape)
        before_return = avg_result_MLP+max_result_MLP

        #unsqueeze
        before_return = avg_result.unsqueeze(2).unsqueeze(3)
        return F.sigmoid(before_return)
        #Shared MLP

    def attention_block(self,features):
        channel_out = self.attention_block_channel(features)
        spatial_out = self.attention_block_spatial(features)
        if channel_out.size(1) != 64:
            print("channel block의 크기가 안맞습니다.")
            exit(0)
        if spatial_out.size(2) != self.image_height or spatial_out.size(3) != self.image_width:
            print("spatial block의 channel 사이즈가 안맞습니다.")
            exit(0)
        broad_channel_out = channel_out.expand(channel_out.size(0),channel_out.size(1),self.image_height,self.image_width)
        if broad_channel_out.size(2) != self.image_height or broad_channel_out.size(3) != self.image_width:
            print("브로드 케스트된 채널의 featuremap 사이즈가 안맞습니다.")
            exit(0)
        broad_spatial_out = spatial_out.expand(channel_out.size(0),channel_out.size(1),self.image_height,self.image_width)
        if broad_spatial_out.size(1) != 64:
            print("브로드 케스트된 channel크기가 안맞습니다.")
            exit(0)
        M_F = broad_channel_out+broad_spatial_out
        return_feature = features+features*M_F
        if return_feature.size(1) != 64 or (return_feature.size(2) != self.image_height or return_feature.size(3) != self.image_width):
            print("출력 채널의 사이즈가 이상합니다.")
            exit(0)
        return return_feature, spatial_out
    def main_forward(self, image):
        after_output = self.base_module_1(image)
        return_feature, spatial_out = self.attention_block(after_output)
        Global_descriptor = F.adaptive_avg_pool2d(self.pooling_module_1(return_feature),(1,1))
        
        if Global_descriptor.size(1) != 512 or (Global_descriptor.size(2) != 1 or Global_descriptor.size(3) != 1):
            print("전역 descriptor의 차원이 이상합니다..")
            exit(0)
        Global_descriptor = Global_descriptor.squeeze(3).squeeze(2)
        return Global_descriptor, spatial_out

    def forward(self, image1,image2):
        global_descriptor1, spatial_out1 = self.main_forward(image1)
        global_descriptor2, spatial_out2 = self.main_forward(image2)
        return global_descriptor1, spatial_out1, global_descriptor2, spatial_out2


class LoopNet_model(nn.Module):#Longterm-dynamic object detection
    def __init__(self,image_width, image_height):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height

        ## origin model
        self.base_module_1 = nn.Sequential()
        self.attention_block_spatial = nn.Sequential()#주 목적은 장면내에서 이미지 픽셀 크기만하게 얻어야되기 때문에 확장되는 구조가 필요함.
        self.attention_block_channel = nn.Sequential()#장면의 특성을 요약하는 역할 
        self.pooling_module_1 = nn.Sequential()#global average pooling도 필요하다.
    
    def forward(self, image):

        return 

