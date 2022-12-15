import numpy as np
import matplotlib.pyplot as plt
import src.dataset as dataset
import src.model as model
import src.loss as loss
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from src.visualize import save_spatial_out,createFolder
from datetime import datetime
import time

def validate_model(epoch, arg_model,validset,loss_ctr,loss_obj,device,weight_value):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    total = 0                                                 
    running_loss = 0.0                                        
    arg_model.eval()
    valid_dataloader = DataLoader(validset,1,shuffle=False,num_workers=8)
    total_2_loss =0
    with torch.no_grad():
        for idx , (query_images, compare_images, labels) in enumerate(valid_dataloader):
            query_images = query_images.to(device)
            compare_images = compare_images.to(device)
            labels = labels.to(device)
            gd_query, spatial_query, gd_compare, spatial_compare = arg_model(query_images, compare_images)
            loss_1 = loss_ctr(gd_query,gd_compare,labels)
            loss_2 = loss_obj(query_images, spatial_query,compare_images, spatial_compare)
            final_loss = loss_1+loss_2*weight_value
            running_loss = running_loss + final_loss.item() 
            total_2_loss = total_2_loss +loss_2.item()
            total +=labels.size(0)         
            del gd_query,spatial_query,gd_compare,spatial_compare, loss_1, loss_2, final_loss     
    mean_val_loss = ( running_loss/total )                            
    print('epoch : ',epoch,'Validation Loss : {:5f}'.format(mean_val_loss),'loss_2 : {:5f}'.format(total_2_loss/total))
    print("valid set : hard positive, negative mining...")
    validset.remake_data(arg_model)
    return mean_val_loss

def train_model(kitti_color_base_path,kitti_color_pose_path, train_data_names,validation_data_names,gamma,dilation_rate,arg_learning_rate,
arg_batch_size, arg_contrastive_margin,max_epoch,weight_value,bad_neighbor,stable_epoch,loss1type,loss1distancetype):
    createFolder("./runs")
    createFolder("./model_weight")
    createFolder("./spatial_result")
    createFolder("./triplet")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    image_width =1240
    image_height =360
    trainset = dataset.useful_dataset(train_data_names,image_height,image_width,bad_neighbor,kitti_color_base_path,kitti_color_pose_path)
    validset = dataset.useful_dataset(validation_data_names,image_height,image_width,bad_neighbor,kitti_color_base_path,kitti_color_pose_path)
    # you can change the size of validset : validset.data_size =140
    validset.data_size =140
    run_now = datetime.now()
    new_path =run_now.strftime('%m-%d_%H_%M_%S')
    createFolder("./runs/"+new_path)
    createFolder("./model_weight/"+new_path)
    
    writter = SummaryWriter("runs/"+new_path)
    LDOD_model = model.LoopNet_model(image_width=image_width,image_height=image_height,gamma=gamma,dilation_rate=dilation_rate).to(device)
    #손실함수
    loss_ctr = loss.SimCLR_Loss(margin=arg_contrastive_margin,type=loss1type,distance=loss1distancetype) 
    loss_obj = loss.obj_Loss()


    #20epoch시 learning rate 증가가 없으면, 그냥 stop
    Optimizer = torch.optim.Adam(LDOD_model.parameters(),lr=arg_learning_rate,weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Optimizer,verbose=True)
    earlystop = loss.EarlyStopping(tolerance=20,min_delta=0)
    best_valid_loss = 100
    # start_epoch=time.time()
    for epoch in range(max_epoch):
        sum_train_loss =0.0
        total =0
        # 초기에는 임의의 trainset을 사용합니다.
        train_dataloader = DataLoader(trainset,arg_batch_size,shuffle=True,num_workers=8)
        start_time = time.time()
        edge_weight =0
        if epoch>stable_epoch:
            edge_weight = weight_value
        for idx , (query_images, compare_images, labels) in enumerate(train_dataloader):
            LDOD_model.zero_grad()
            query_images = query_images.to(device)
            compare_images = compare_images.to(device)
            labels = labels.to(device)

            gd_query, spatial_query, gd_compare, spatial_compare = LDOD_model(query_images, compare_images)
            loss_1 = loss_ctr(gd_query,gd_compare,labels)
            loss_2 = loss_obj(query_images, spatial_query,compare_images, spatial_compare)
            final_loss = loss_1+loss_2*edge_weight
            sum_train_loss = sum_train_loss + final_loss.item() 
            total +=labels.size(0)
            final_loss.backward()
            Optimizer.step()
            del gd_query,spatial_query,gd_compare,spatial_compare, loss_1, loss_2, final_loss
        #validation set
        end_time = time.time()
        mean_train_loss = sum_train_loss/total
        print("mean train loss : {:.5f},  time consume : {:.1f}".format(mean_train_loss,end_time-start_time))
        val_loss = validate_model(epoch+1, LDOD_model,validset,loss_ctr,loss_obj,device,edge_weight)
        #save_loss
        writter.add_scalar("loss/train",mean_train_loss,epoch+1)
        writter.add_scalar("loss/valid",val_loss,epoch+1)
        folder_name = "spatial_result/"+\
                    new_path+\
                    "arg_batch_size_"+\
                    str(arg_batch_size)+\
                    "-weight_value_"+\
                    str(weight_value)+\
                    "-dilation_rate_"+\
                    str(dilation_rate)+\
                    "-gamma_"+\
                    str(16)

        save_spatial_out(LDOD_model,[20,30,40,80],validset,folder_name,str(epoch+1))
        
        print("train set : hard positive, negative mining...")
        trainset.remake_data(LDOD_model)
        if best_valid_loss>val_loss:
            best_valid_loss =val_loss
            #save_model
            model_name = new_path+"/-arg_batch_size_"+\
                        str(arg_batch_size)+\
                        "-weight_value_"+\
                        str(weight_value)+\
                        "-dilation_rate_"+\
                        str(dilation_rate)+\
                        "-gamma_"+\
                        str(gamma)+\
                        "-epoch_"+\
                        str(epoch+1)+\
                        "_weight.pth"
            torch.save(LDOD_model.state_dict(),"./model_weight/"+model_name)
            #save_spatial_out_of_query
        
        if epoch>3:
            scheduler.step(val_loss)
        LDOD_model.train()
        if earlystop.test(val_loss):
            print("early stop (",epoch+1,")")
            break
    writter.close()
    del LDOD_model

    
    
