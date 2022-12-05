import numpy as np
import matplotlib.pyplot as plt
import src.dataset as dataset
import src.model as model
import src.loss as loss
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from src.visualize import save_spatial_out


# valid = dataset.useful_dataset(valid_group)
def validate_model(epoch, arg_model,validset,loss_ctr,loss_obj,device,weight_value):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    total = 0                                                 
    running_loss = 0.0                                        
    arg_model.eval()
    with torch.no_grad():
        for idx , (query_images, compare_images, labels) in enumerate(validset):
            # loss = criterion(outputs, labels)                 
            # _, predicted = torch.max(outputs.data, 1)         
            # total += labels.size(0)                           
            # correct += (predicted == labels).sum().item()     
            # running_loss = running_loss + loss.item() 
            query_images = query_images.to(device)
            compare_images = compare_images.to(device)
            labels = labels.to(device)
            gd_query, spatial_query, gd_compare, spatial_compare = arg_model(query_images, compare_images)
            loss_1 = loss_ctr(gd_query,gd_compare,labels)
            loss_2 = loss_obj(spatial_query,spatial_compare)
            final_loss = loss_1+loss_2*weight_value
            running_loss = running_loss + final_loss.item() 
            total +=labels.size(0)         
            del gd_query,spatial_query,gd_compare,spatial_compare, loss_1, loss_2, final_loss     
    mean_val_loss = ( running_loss/total )                            
    print('epoch : ',epoch,'Validation Loss : ',mean_val_loss )
    return mean_val_loss

def train_model(train_data_names,validation_data_names,gamma,dilation_rate,arg_learning_rate,arg_batch_size, arg_contrastive_margin,max_epoch,weight_value):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    image_width =1240
    image_height =370
    trainset = dataset.useful_dataset(train_data_names,image_height,image_width)
    validset = dataset.useful_dataset(validation_data_names,image_height,image_width)
    print("훈련데이터셋의 수 :",len(trainset))
    print("평가데이터셋의 수 :",len(validset))
    train_dataloader = DataLoader(trainset,arg_batch_size,shuffle=True,num_workers=4)
    valid_dataloader = DataLoader(validset,1,shuffle=False,num_workers=4)
    writter = SummaryWriter("runs")
    
    LDOD_model = model.LDOD_model(image_width=image_width,image_height=image_height,gamma=gamma,dilation_rate=dilation_rate).to(device)
    #손실함수
    loss_ctr = loss.SimCLR_Loss(margin=arg_contrastive_margin) 
    loss_obj = loss.obj_Loss(kernel_size=3)


    #20epoch시 learning rate증가가 없으면, 그냥 stop
    Optimizer = torch.optim.Adam(LDOD_model.parameters(),lr=arg_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Optimizer,verbose=True)
    earlystop = loss.EarlyStopping(tolerance=20,min_delta=0)
    
    print("학습을 시작합니다.")
    for epoch in range(max_epoch):
        sum_train_loss =0.0
        total =0
        for idx , (query_images, compare_images, labels) in enumerate(train_dataloader):
            LDOD_model.zero_grad()
            query_images = query_images.to(device)
            compare_images = compare_images.to(device)
            labels = labels.to(device)

            gd_query, spatial_query, gd_compare, spatial_compare = LDOD_model(query_images, compare_images)
            loss_1 = loss_ctr(gd_query,gd_compare,labels)
            loss_2 = loss_obj(spatial_query,spatial_compare)
            final_loss = loss_1+loss_2*weight_value
            sum_train_loss = sum_train_loss + final_loss.item() 
            total +=labels.size(0)
            final_loss.backward()
            Optimizer.step()
            del gd_query,spatial_query,gd_compare,spatial_compare, loss_1, loss_2, final_loss
        #validation set
        mean_train_loss = sum_train_loss/total
        
        val_loss = validate_model(epoch+1, LDOD_model,valid_dataloader,loss_ctr,loss_obj,device,weight_value)

        #save_loss
        writter.add_scalar("loss/train",mean_train_loss,epoch+1)
        writter.add_scalar("loss/valid",val_loss,epoch+1)

        
        if (epoch+1)%10 ==0 or epoch ==0:
            #save_model
            model_name = "-arg_batch_size_"+\
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
            folder_name = "spatial_result/"+\
                        "arg_batch_size_"+\
                        str(arg_batch_size)+\
                        "-weight_value_"+\
                        str(weight_value)+\
                        "-dilation_rate_"+\
                        str(dilation_rate)+\
                        "-gamma_"
            save_spatial_out(LDOD_model,[100,200,300,400],valid_dataloader,folder_name,str(epoch+1))

        scheduler.step(val_loss)
        LDOD_model.train()
        if earlystop.test(val_loss):
            print("early stop (",epoch+1,")")
            break
    writter.close()
    del LDOD_model

    
    