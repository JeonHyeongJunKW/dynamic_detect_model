import src.model as model
import src.dataset as dataset
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from src.visualize import *
import time
import glob
import argparse

def view_all_sequence(model_path, dataset_path, image_width, image_height):
    LDOD_model = model.LoopNet_model(image_width=image_width,image_height=image_height,gamma=16,dilation_rate=4).to(device)
    PATH = model_path
    LDOD_model.load_state_dict(torch.load(PATH))
    LDOD_model.eval()
    before_name = dataset_path
    data_file_name = glob.glob(before_name+"/*")
    data_file_name.sort()
    dataset_size = len(data_file_name)
    transform = transforms.Compose(
        [transforms.ToTensor(),\
        transforms.Resize((image_height,image_width)),\
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
        )
    with torch.no_grad():
        for idx in range(dataset_size):
            query_name = data_file_name[idx]
            query_image = cv2.imread(query_name,cv2.IMREAD_COLOR)
            torch_query = transform(query_image).cuda().unsqueeze(0)
            gd_query, spatial_query = LDOD_model.main_forward(torch_query)
            spatial_image = (spatial_query[0].cpu().permute(1,2,0).detach().numpy()*255).astype(np.uint8)
            spatial_image = np.repeat(spatial_image,3,2)
            spatial_image = cv2.resize(spatial_image, dsize=(image_width, image_height), interpolation=cv2.INTER_LINEAR)
            query_image = cv2.resize(query_image, dsize=(image_width, image_height), interpolation=cv2.INTER_LINEAR)
            blending_image = cv2.addWeighted(query_image,0.5,spatial_image,0.5,0)
            result = np.vstack((query_image,spatial_image,blending_image))
            result = cv2.resize(result, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("blending",result)
            cv2.waitKey(10)
            del gd_query,spatial_query

parser = argparse.ArgumentParser(description='dynamic object detection model test')
parser.add_argument('--model_path', type=str,default="main.pth")
parser.add_argument('--dataset_path', type=str,default="/media/jeon/hard/Kitti_dataset/dataset/sequences/00/image_2")
parser.add_argument('--image_height', type=int,default=360)
parser.add_argument('--image_width', type=int,default=1240)
args = parser.parse_args()
# parser.imag
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
view_all_sequence(args.model_path,args.dataset_path,args.image_width,args.image_height)#1240,360
