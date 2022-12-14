import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import torch
import math

class attention_block(nn.Module):
    def __init__(self,in_channel,gamma,dilation_rate):
        super().__init__()
        self.gamma = gamma
        self.inner_channel = int(in_channel/self.gamma)
        self.dilation_rate =dilation_rate
        self.spatial = nn.Sequential(Conv2d(in_channel,self.inner_channel,1,padding=0),
                                                    Conv2d(self.inner_channel,self.inner_channel,3,dilation=self.dilation_rate,padding=4),
                                                    Conv2d(self.inner_channel,self.inner_channel,3,dilation=self.dilation_rate,padding=4),
                                                    Conv2d(self.inner_channel,1,1,padding=0))#마지막에 0과 1사이의 값을 가진다.
        self.shared_MLP = nn.Sequential(nn.Linear(in_channel,self.inner_channel),
                                        nn.ReLU(),
                                        nn.Linear(self.inner_channel,in_channel))
    def attention_block_channel(self, featuremap):
        avg_result = F.adaptive_avg_pool2d(featuremap,(1,1))# output : [10, 64, 1, 1]
        max_result = F.adaptive_max_pool2d(featuremap,(1,1))

        avg_result = avg_result.squeeze(3).squeeze(2)
        max_result = max_result.squeeze(3).squeeze(2)

        avg_result_MLP = self.shared_MLP(avg_result)
        max_result_MLP = self.shared_MLP(max_result)
        before_return = avg_result_MLP+max_result_MLP

        before_return = avg_result.unsqueeze(2).unsqueeze(3)
        return torch.sigmoid(before_return)

    def forward(self, featuremap):
        channel_out = torch.sigmoid(self.attention_block_channel(featuremap))# output : [10, 64, 1, 1]# output : 0~1
        spatial_out = torch.sigmoid(self.spatial(featuremap))# output : 0~1
        broad_channel_out = channel_out.expand(channel_out.size(0),channel_out.size(1),featuremap.size(2),featuremap.size(3))# output : 0~1
        broad_spatial_out = spatial_out.expand(channel_out.size(0),channel_out.size(1),featuremap.size(2),featuremap.size(3))# output : 0~1
        M_F = broad_channel_out+broad_spatial_out# output : 0~2
        return_feature = featuremap+featuremap*M_F
        return return_feature, spatial_out

class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]#입력된 사이즈
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

class LoopNet_model(nn.Module):#Longterm-dynamic object detection
    def __init__(self,image_width, image_height,gamma=16,dilation_rate=4):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.gamma = gamma
        self.dilation_rate =dilation_rate
        ## origin model
        self.base_module_1 = nn.Sequential(Conv2d(3,64,7,padding=3),
                                            Conv2d(64,64,5,padding=2))
        self.base_module_1_half = nn.Sequential(Conv2d(3,64,7,padding=3),
                                            Conv2d(64,64,5,padding=2))
        self.base_module_2 = nn.Sequential(Conv2d(64,128,3,padding=1),
                                            Conv2d(128,128,3,padding=1))
        self.base_module_2_quarter = nn.Sequential(Conv2d(3,64,3,padding=1),
                                            Conv2d(64,128,3,padding=1))
        self.base_module_3 = nn.Sequential(Conv2d(128,256,3,padding=1),
                                            Conv2d(256,256,3,padding=1))
        self.base_module_4 = nn.Sequential(Conv2d(256,512,3,padding=1),
                                            Conv2d(512,512,3,padding=1))

        self.attention_module_1 =attention_block(64,self.gamma,self.dilation_rate)
        self.attention_module_1_half =attention_block(64,self.gamma,self.dilation_rate)
        self.attention_module_2 =attention_block(128,self.gamma,self.dilation_rate)
        self.attention_module_2_quarter =attention_block(128,self.gamma,self.dilation_rate)
        self.attention_module_3 =attention_block(256,self.gamma,self.dilation_rate)
        self.attention_module_4 =attention_block(512,self.gamma,self.dilation_rate)

        self.spatial_pooling_1 =PyramidPooling([4,2,1],"avg")
        self.spatial_pooling_2 =PyramidPooling([2,1],"avg")
        self.spatial_pooling_3 =PyramidPooling([1],"avg")
        #주 목적은 장면내에서 이미지 픽셀 크기여야한다.

    def main_forward(self, image):
        image_half = F.interpolate(image,size=(int(self.image_height/2),int(self.image_width/2)),mode='bicubic',align_corners=False)
        image_quarter = F.interpolate(image,size=(int(self.image_height/4),int(self.image_width/4)),mode='bicubic',align_corners=False)

        rf_1, _ = self.attention_module_1(self.base_module_1(image))#출력이 
        rf_1_half, _ = self.attention_module_1_half(self.base_module_1(image_half))
        start_rf_2 = torch.max_pool2d(rf_1,2)+rf_1_half

        rf_2, _ = self.attention_module_2(self.base_module_2(start_rf_2))
        rf_2_half, _ = self.attention_module_2_quarter(self.base_module_2_quarter(image_quarter))
        start_rf_3 = torch.max_pool2d(rf_2,2)+rf_2_half

        rf_3, spatial_out = self.attention_module_3(self.base_module_3(start_rf_3))
        start_rf_4 = torch.max_pool2d(rf_3,2)

        rf_4, _ = self.attention_module_4(self.base_module_4(start_rf_4))
        end_rf_4 = torch.max_pool2d(rf_4,2)

        sp_result_1 = self.spatial_pooling_1(start_rf_3)
        sp_result_2 = self.spatial_pooling_2(start_rf_4)
        sp_result_3 = self.spatial_pooling_3(end_rf_4)
        Global_descriptor = (torch.cat((sp_result_1,sp_result_2,sp_result_3),1))
        #Normalize
        Global_descriptor = Global_descriptor/torch.norm(Global_descriptor,dim=1,keepdim=True)
        return Global_descriptor, spatial_out
    
    def forward(self, image1,image2):
        global_descriptor1, spatial_out1 = self.main_forward(image1)
        global_descriptor2, spatial_out2 = self.main_forward(image2)
        return global_descriptor1, spatial_out1, global_descriptor2, spatial_out2 

