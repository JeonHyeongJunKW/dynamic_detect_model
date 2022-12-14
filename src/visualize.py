import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.dataset import useful_dataset
from torch.utils.data import DataLoader

def tensor2Colorimage(tensor_image):
    Mean_c = 0.5
    SD_c = 0.5
    output_image = ((tensor_image.cpu().permute(1,2,0).detach().numpy()*SD_c+Mean_c)*255).astype(np.uint8)
    clipped_image = np.clip(output_image,0,255)
    return clipped_image

def tensor2Grayimage(tensor_image):
    Mean_c = 0.5
    SD_c = 0.5
    output_image = ((tensor_image.cpu().permute(1,2,0).squeeze(2).detach().numpy()*SD_c+Mean_c)*255).astype(np.uint8)
    clipped_image = np.clip(output_image,0,255)
    return clipped_image

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def save_spatial_out(model,query_idxs,validset,folder,sub_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')                                      
    model.eval()
    valid_dataloader = DataLoader(validset,1,shuffle=False,num_workers=4)
    with torch.no_grad():
        for idx , (query_images, compare_images, labels) in enumerate(valid_dataloader):
            if idx in query_idxs:
                query_images = query_images.to(device)
                compare_images = compare_images.to(device)
                labels = labels.to(device)
                gd_query, spatial_query, gd_compare, spatial_compare = model(query_images, compare_images)
                #print(spatial_query)
                spatial_image = (spatial_query[0].cpu().permute(1,2,0).squeeze(2).detach().numpy()*255).astype(np.uint8)
                query_image = tensor2Colorimage(query_images[0])
                createFolder(folder)
                createFolder(folder+"/"+sub_folder)
                cv2.imwrite(folder+"/"+sub_folder+"/"+str(idx)+"_query.png",query_image)
                cv2.imwrite(folder+"/"+sub_folder+"/"+str(idx)+"_spatial.png",spatial_image)
                del gd_query,spatial_query,gd_compare,spatial_compare
