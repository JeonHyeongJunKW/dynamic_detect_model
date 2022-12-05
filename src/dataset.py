import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import cv2
import random
dataset_base_path = "/media/jeon/hard/Kitti_dataset/dataset_gray/"
d_g_pose_path = dataset_base_path+"poses/"
d_g_image_path = dataset_base_path+"sequences/"
d_g_camera_name = "image_0/"



class useful_dataset(Dataset):
    def __init__(self,data_group,image_height,image_width):
        # print("데이터셋을 생성합니다.")
        self.out_negative_range = 25#m
        self.in_positive_range =10#m
        self.query_btw_dist =1#m
        self.loop_closure_query_dist =30#frames
        self.bad_neighbor_number =20

        self.data_group =[]
        self.data_size =0
        self.data_start_idx =[]
        self.seq_start_idx = 0
        self.seq_names =data_group
        self.transform = transforms.Compose(
        [transforms.ToTensor(),\
        transforms.Resize((image_height,image_width)),\
        transforms.Normalize(0.5, 0.5) ]
        )
        for data_member in self.seq_names:
            self.data_start_idx.append(self.seq_start_idx)
            file_name =d_g_pose_path+data_member+".txt"
            f = open(file_name,'r')
            lines = f.readlines()
            one_group_query = []
            one_loop_query =[]
            # print(data_member,"sequence는 ", len(lines),"개의 데이터가 있습니다.")
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
                
                loopclosure_candidates = list(range(0,one_group_query[lower_bound_query]))+list(range(one_group_query[upper_bound_query],count-1))
                # print(query_idx, lower_bound_query,upper_bound_query)
                if real_idx in loopclosure_candidates:
                    print("잘못된 표본그룹입니다.")
                    exit(0)
                #후보들을 좌표로 변환합니다.
                candidate_pose =[[0,0] for _ in range(len(loopclosure_candidates))]
                neighhbor = NearestNeighbors(n_neighbors = 1,radius=self.in_positive_range)
                #변화된 좌표로부터 knn알고리즘을 사용하여 self.in_positive_range 범위에 안에 해당 점이 존재하는지 확인합니다.
                
                for pose_idx, candidate_idx in enumerate(loopclosure_candidates):
                    candidate_pose[pose_idx] = real_xz[candidate_idx]
                neighhbor.fit(candidate_pose)

                neighbors_dist, neighbors_idx = neighhbor.kneighbors([real_xz[real_idx]])
                
                positive_idx =0
                #query와 positive 그룹을 구합니다.
                if neighbors_dist[0][0] <= self.in_positive_range:
                    positive_idx = loopclosure_candidates[neighbors_idx[0][0]]#원래 인덱스로 바꿉니다.
                    loop_query_number+=1
                else:
                    continue
                #negative 그룹을 구합니다. 초기에는 랜덤으로 뽑는걸로 구합니다. 이건 비슷한군을 찾아서 업데이트하는걸로 해야할듯 초반에는 랜덤으로 뽑는형태로 하자.
                bad_neighhbor = NearestNeighbors()
                bad_neighhbor.fit(candidate_pose)
                nontrival_positive = bad_neighhbor.radius_neighbors([real_xz[real_idx]],radius=self.out_negative_range,return_distance=False)
                real_bad_neighbor = np.setdiff1d(np.array(loopclosure_candidates),np.array(nontrival_positive[0]),assume_unique=True)
                #query그룹을 결성하여 넣습니다.
                bad_idx = np.random.choice(real_bad_neighbor,self.bad_neighbor_number,replace=False)
                one_loop_query.append([real_idx,positive_idx,bad_idx.tolist()])
                
            self.seq_start_idx += loop_query_number
            self.data_size =self.seq_start_idx
            self.data_group.append(one_loop_query)
            # print("query는 ",count,"개입니다.")
            # print("loop query는 ",loop_query_number,"입니다.")
            # visualize.view_query_and_db(one_loop_query,\
            #     d_g_image_path+data_member+"/"+d_g_camera_name)

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
        query_name = d_g_image_path+self.seq_names[real_seq]+"/"+d_g_camera_name+'{0:06d}'.format(query_idx)+'.png'
        positive_name = d_g_image_path+self.seq_names[real_seq]+"/"+d_g_camera_name+'{0:06d}'.format(positive_idx)+'.png'
        negative_names =[d_g_image_path+self.seq_names[real_seq]+"/"+d_g_camera_name+'{0:06d}'.format(k)+'.png' for k in negative_idxs]

        query_image = cv2.imread(query_name,cv2.IMREAD_GRAYSCALE)
        query_image = self.transform(query_image)
        if label ==1:#positive
            positive_image =cv2.imread(positive_name,cv2.IMREAD_GRAYSCALE)
            positive_image = self.transform(positive_image)
            return query_image, positive_image, label
        else :
            negative_idx =random.randint(0,len(negative_names)-1)
            negative_image = cv2.imread(negative_names[negative_idx],cv2.IMREAD_GRAYSCALE)
            negative_image = self.transform(negative_image)
            return query_image, negative_image, label
        
    
    def remake_data(self):
        #descriptor의 형태를 고려하여 재정의합니다.
        pass
    
    def __len__(self):
        return self.data_size