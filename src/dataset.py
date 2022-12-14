import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import cv2
import random
import torch
import time
import math



class useful_dataset(Dataset):
    def __init__(self,data_group,
                    image_height,
                    image_width,
                    bad_neighbor,
                    kitti_color_base_path,
                    kitti_color_pose_path):
        self.out_negative_range = 25#m
        self.in_positive_range =10#m
        self.query_btw_dist =1#m
        self.d_g_camera_name = "image_2/"
        self.loop_closure_query_dist =30#frames
        self.bad_neighbor_number =bad_neighbor
        self.d_g_pose_path =kitti_color_pose_path#kitti_color_pose_path
        self.d_g_image_path =kitti_color_base_path
        self.data_group =[]
        self.data_group_size =[]
        self.data_size =0
        self.data_start_idx =[]
        self.seq_start_idx = 0
        self.seq_names =data_group
        self.transform = transforms.Compose(
        [transforms.ToTensor(),\
        transforms.Resize((image_height,image_width)),\
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]#s
        )
        for data_member in self.seq_names:
            self.data_start_idx.append(self.seq_start_idx)
            file_name =self.d_g_pose_path+data_member+".txt"
            f = open(file_name,'r')
            lines = f.readlines()
            dataset_size = len(lines)
            self.data_group_size.append(dataset_size)
            one_group_query = []
            one_loop_query =[]
            print(data_member,"sequence는 ", len(lines),"개의 데이터가 있습니다.")
            last_x =0
            last_z =0
            count =0
            real_xz = []
            #위치 정보 라인별로 검사합니다. 
            for line_idx, line in enumerate(lines):
                params = line.split(" ")
                x = float(params[3])
                z = float(params[11])
                real_xz.append([x,z])
                if line_idx ==0:
                    last_x =x
                    last_z =z
                    continue
                else:
                    dist = (x-last_x)*(x-last_x)+(z-last_z)*(z-last_z)
                    if dist>self.query_btw_dist:#1m 차이나면 
                        count+=1
                        last_x =x
                        last_z =z
                        one_group_query.append(line_idx)# query로써 index를 저장해둡니다.
            #각 query에서 일정간격차이가 나면서 거리가 가까운 데이터셋을 찾습니다. 
            loop_query_number =0
            for query_idx, real_idx in enumerate(one_group_query):
                # 최소최대 간격입니다.
                lower_bound_query = query_idx-int(self.loop_closure_query_dist/2) if query_idx-int(self.loop_closure_query_dist/2)>=0 else 0
                upper_bound_query = query_idx+int(self.loop_closure_query_dist/2) if query_idx+int(self.loop_closure_query_dist/2)<count else count-1
                # 리스트에 이 영역에 속하는 데이터군을 넣어둡니다.
                
                # 현재프레임이랑 떨어진 프레임들을 구합니다. 
                loopclosure_candidates = list(range(0,one_group_query[lower_bound_query]))+list(range(one_group_query[upper_bound_query],count-1))
                if real_idx in loopclosure_candidates:
                    print("잘못된 표본그룹입니다.")
                    exit(0)
                #후보들을 좌표로 변환합니다.
                candidate_pose =[[0,0] for _ in range(len(loopclosure_candidates))]
                
                ##positive를 구합니다.
                #변화된 좌표로부터 knn알고리즘을 사용하여 self.in_positive_range 범위에 안에 해당 점이 존재하는지 확인합니다.
                neighhbor = NearestNeighbors(radius=self.in_positive_range,n_jobs=-1)
                for pose_idx, candidate_idx in enumerate(loopclosure_candidates):
                    candidate_pose[pose_idx] = real_xz[candidate_idx]
                neighhbor.fit(candidate_pose)

                neighbors_dist, neighbors_idx = neighhbor.radius_neighbors([real_xz[real_idx]])
                if len(neighbors_idx[0])==0:
                    continue
                positive_idx =0
                #query와 초기 positive와 positive 그룹을 구합니다.
                positive_idxes =[ loopclosure_candidates[value] for value in neighbors_idx[0]]
                if neighbors_dist[0][0] <= self.in_positive_range:
                    positive_idx = positive_idxes[0]#원래 인덱스로 바꿉니다.
                    loop_query_number+=1
                else:
                    continue

                #negative 그룹을 구합니다. 초기에는 랜덤으로 뽑는걸로 구합니다. 이건 비슷한군을 찾아서 업데이트하는걸로 해야할듯 초반에는 랜덤으로 뽑는형태로 하자.
                bad_candidate_id = np.setdiff1d(np.array(loopclosure_candidates),np.array(positive_idxes),assume_unique=True)
                bad_candidate_pose =[real_xz[id] for id in bad_candidate_id]
                bad_neighhbor = NearestNeighbors(n_jobs=-1)
                bad_neighhbor.fit(bad_candidate_pose)
                
                #25m 밖의 이웃들 기존 bad_candidate_id에서 제외합니다.
                nontrival_positive = bad_neighhbor.radius_neighbors([real_xz[real_idx]],radius=self.out_negative_range,return_distance=False)
                
                real_nontrival_neighbor = [ bad_candidate_id[real_sim_bad_idx] for real_sim_bad_idx in nontrival_positive[0]]
                real_bad_neighbor = np.setdiff1d(np.array(bad_candidate_id),np.array(real_nontrival_neighbor),assume_unique=True)
                for qu_po_idx in positive_idxes:
                    if qu_po_idx in real_bad_neighbor:
                        print("잘못된 분류")
                        exit()
                #query그룹을 결성하여 넣습니다.
                bad_idx = np.random.choice(real_bad_neighbor,self.bad_neighbor_number,replace=False)
                one_loop_query.append([real_idx,positive_idx,bad_idx.tolist(),real_bad_neighbor,positive_idxes])
                
            self.seq_start_idx += loop_query_number
            self.data_size =self.seq_start_idx
            self.data_group.append(one_loop_query)
            print("query는 ",self.data_size,"개입니다.")
    def remake_data(self, model):
        #descriptor의 형태를 고려하여 재정의합니다.

        model.eval()
        #이미지들을 descriptor를 embedding으로 변환합니다.
        saved_reps =[[] for i in range(len(self.data_group))]
        with torch.no_grad():
            for seq_idx in range(len(self.data_group)):
                #20개정도씩 쌓은다음에 출력얻고, 나눠서 저장
                #나머지 연산으로 남은 특징점들은 배치만들어서 저장
                datasize = self.data_group_size[seq_idx]
                convert_batch_size =20
                convert_batch_number = math.floor(datasize/convert_batch_size)
                residul_number = datasize-convert_batch_number*convert_batch_size
                for query_set in range(convert_batch_number):
                    query_images =[]
                    for query_read_idx in range(convert_batch_size):
                        real_read_idx = query_set*convert_batch_size+ query_read_idx
                        query_name = self.d_g_image_path+self.seq_names[seq_idx]+"/"+self.d_g_camera_name+'{0:06d}'.format(real_read_idx)+'.png'
                        query_image = cv2.imread(query_name,cv2.IMREAD_COLOR)
                        query_image = self.transform(query_image)
                        query_images.append(query_image)
                    query_images = torch.stack(query_images).cuda()
                    embedding_result, _ = model.main_forward(query_images)
                    emb_rep = embedding_result.cpu().detach().numpy()
                    for idx in range(convert_batch_size):
                        saved_reps[seq_idx].append(emb_rep[idx])
                query_images =[]
                for resid_idx in range(residul_number):
                    real_read_idx = convert_batch_number*convert_batch_size + resid_idx
                    query_name = self.d_g_image_path+self.seq_names[seq_idx]+"/"+self.d_g_camera_name+'{0:06d}'.format(real_read_idx)+'.png'
                    query_image = cv2.imread(query_name,cv2.IMREAD_COLOR)
                    query_image = self.transform(query_image)
                    query_images.append(query_image)
                query_images = torch.stack(query_images).cuda()
                embedding_result, _ = model.main_forward(query_images)
                emb_rep = embedding_result.cpu().detach().numpy()
                for idx in range(residul_number):
                    saved_reps[seq_idx].append(emb_rep[idx])
                if len(saved_reps[seq_idx]) != datasize:
                    print("배치 수식에 오류가 있습니다.")
        #거리비교를 하여 hard negative를 할당합니다. 
        for seq_idx in range(len(self.data_group)):
            for query_idx in range(len(self.data_group[seq_idx])):
                query_set = self.data_group[seq_idx][query_idx]
                potential_negative = query_set[3]#이 descriptor를 가져오고, 가까운 descriptor를 얻어온다.
                pn_descriptors = [ saved_reps[seq_idx][idx] for idx in potential_negative]#hard negative 후보들의 descriptor
                neigh = NearestNeighbors(n_neighbors=self.bad_neighbor_number,n_jobs=-1).fit(pn_descriptors)
                neigh_idxs = neigh.kneighbors([saved_reps[seq_idx][query_set[0]]],return_distance = False)#x와 유사한 descriptor
                new_negatives = neigh_idxs[0].tolist()
                if len(new_negatives) != self.bad_neighbor_number:
                    print(len(new_negatives))
                    print("형태가 틀렸습니다.")
                real_hards = [potential_negative[neg_idx] for neg_idx in new_negatives]
                self.data_group[seq_idx][query_idx][2] = real_hards

                #positive data를 뽑습니다.
                positives = query_set[4]
                pp_descriptors = [ saved_reps[seq_idx][idx] for idx in positives]#hard negative 후보들의 descriptor
                neigh = NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(pp_descriptors)
                neigh_idxs = neigh.kneighbors([saved_reps[seq_idx][query_set[0]]],return_distance = False)#x와 유사한 descriptor
                best_positive = neigh_idxs[0].tolist()[0]
                self.data_group[seq_idx][query_idx][1]=positives[best_positive]
                ## query 마다 저장해주세요.
                query_name = self.d_g_image_path+self.seq_names[seq_idx]+"/"+self.d_g_camera_name+'{0:06d}'.format(query_set[0])+'.png'
                save_query_image = cv2.imread(query_name,cv2.IMREAD_COLOR)
                positive_name = self.d_g_image_path+self.seq_names[seq_idx]+"/"+self.d_g_camera_name+'{0:06d}'.format(self.data_group[seq_idx][query_idx][1])+'.png'
                positive_image = cv2.imread(positive_name,cv2.IMREAD_COLOR)
                negative_name = self.d_g_image_path+self.seq_names[seq_idx]+"/"+self.d_g_camera_name+'{0:06d}'.format(self.data_group[seq_idx][query_idx][2][0])+'.png'
                negative_image = cv2.imread(negative_name,cv2.IMREAD_COLOR)
                cv2.imwrite("./triplet/"+self.seq_names[seq_idx]+"/"+str(query_set[0])+"_1query.png",save_query_image)
                cv2.imwrite("./triplet/"+self.seq_names[seq_idx]+"/"+str(query_set[0])+"_2positive.png",positive_image)
                cv2.imwrite("./triplet/"+self.seq_names[seq_idx]+"/"+str(query_set[0])+"_3negative.png",negative_image)
        model.train()

    def __getitem__(self, index):
        real_seq_idx = index-self.data_start_idx[-1]
        real_seq =len(self.data_start_idx)-1
        for j, start_idx in enumerate(self.data_start_idx):
            if start_idx >index:
                real_seq_idx = index - self.data_start_idx[j-1]
                real_seq  = j-1
                break
        # 실제 쿼리를 가져옵니다.
        data_set = self.data_group[real_seq][real_seq_idx]
        query_idx = data_set[0]
        positive_idx = data_set[1]
        negative_idxs = data_set[2]

        label = random.randint(0,1)
        # 이미지를 불러옵니다.
        query_name = self.d_g_image_path+self.seq_names[real_seq]+"/"+self.d_g_camera_name+'{0:06d}'.format(query_idx)+'.png'
        positive_name = self.d_g_image_path+self.seq_names[real_seq]+"/"+self.d_g_camera_name+'{0:06d}'.format(positive_idx)+'.png'
        negative_names =[self.d_g_image_path+self.seq_names[real_seq]+"/"+self.d_g_camera_name+'{0:06d}'.format(k)+'.png' for k in negative_idxs]

        query_image = cv2.imread(query_name,cv2.IMREAD_COLOR)
        query_image = self.transform(query_image)
        if label ==0:#positive
            positive_image =cv2.imread(positive_name,cv2.IMREAD_COLOR)
            positive_image = self.transform(positive_image)
            return query_image, positive_image, label
        else :
            negative_idx =random.randint(0,len(negative_names)-1)
            negative_image = cv2.imread(negative_names[negative_idx],cv2.IMREAD_COLOR)
            negative_image = self.transform(negative_image)
            return query_image, negative_image, label
        
    
    
    
    def __len__(self):
        return self.data_size