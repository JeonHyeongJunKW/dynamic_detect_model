#train데이터셋에 있는 sequence의 결과를 확인합니다.
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# d_g_image_path = dataset_base_path+"sequences/"
# d_g_camera_name = "image_0/"


##------------------경로를 확인합니다.

def view_groudtruth_pose(path_name,point_2d,color_arg):
    dataset_base_path = "/media/jeon/hard/Kitti_dataset/dataset_gray/"
    d_g_pose_path = dataset_base_path+"poses/"
    file_name =d_g_pose_path+path_name
    f = open(file_name,'r')
    lines = f.readlines()
    path = [[],[]]#x,z값이 나눠서 저장됩니다.
    for idx, line in enumerate(lines):
        #z와 x를 뽑아와서 출력해야함. 즉 4번과 12번임
        params = line.split(" ")
        x = params[3]
        z = params[11]
        path[0].append(float(x))
        path[1].append(float(z))
        point_2d.append([x,z])
    plt.plot(path[0],path[1],color_arg)

def view_train_path():
    color =['r','g','b']
    for color_idx, seq_idx in enumerate(["00.txt","05.txt","06.txt"]):
        point_2d_new = [[],[]]
        plt.subplot(1,3,color_idx+1)
        view_groudtruth_pose(seq_idx,point_2d_new,color[color_idx])
    plt.show()

def view_validation_path():
    color =['r','g','b']
    for color_idx, seq_idx in enumerate(["02.txt","08.txt"]):
        point_2d_new = [[],[]]
        plt.subplot(1,3,color_idx+1)
        view_groudtruth_pose(seq_idx,point_2d_new,color[color_idx])
    plt.show()

def view_query_and_db(image_set,image_dir):

    for images_idx in image_set:
        query_name = image_dir+'{0:06d}'.format(images_idx[0])+'.png'
        
        db_name =image_dir+'{0:06d}'.format(images_idx[1])+'.png'
        
        query_image = cv2.imread(query_name,cv2.IMREAD_COLOR)
        db_image = cv2.imread(db_name,cv2.IMREAD_COLOR)


        query_image = cv2.putText(query_image,'{0:06d}'.format(images_idx[0]),(20,20),cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255), 3, cv2.LINE_AA)
        db_image = cv2.putText(db_image,'{0:06d}'.format(images_idx[1]),(20,20),cv2.FONT_HERSHEY_PLAIN,2,(0, 0, 255), 3, cv2.LINE_AA)
        print(images_idx[0],images_idx[1])
        dddh=cv2.hconcat([query_image,db_image])
        # cv2.imshow("matchd",dddh)
        dst2 = cv2.resize(dddh, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("match",dst2)
        cv2.waitKey(0)

# get_groudtruth_pose(sample_groundtruth_name,path)
# view_train_path()
def tensor2image(tensor_image):
  output_image = ((tensor_image.cpu().permute(1,2,0).detach().numpy()*0.5+0.5)*255).astype(np.int32)
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
    with torch.no_grad():
        for idx , (query_images, compare_images, labels) in enumerate(validset):
            if idx in query_idxs:
                query_images = query_images.to(device)
                compare_images = compare_images.to(device)
                labels = labels.to(device)
                gd_query, spatial_query, gd_compare, spatial_compare = model(query_images, compare_images)
                #print(spatial_query)
                spatial_image = (spatial_query[0].cpu().permute(1,2,0).detach().numpy()*255).astype(np.int32)
                query_image = tensor2image(query_images[0])
                createFolder(folder)
                createFolder(folder+"/"+sub_folder)
                cv2.imwrite(folder+"/"+sub_folder+"/"+str(idx)+"_query.png",query_image)
                cv2.imwrite(folder+"/"+sub_folder+"/"+str(idx)+"_spatial.png",spatial_image)
                del gd_query,spatial_query,gd_compare,spatial_compare