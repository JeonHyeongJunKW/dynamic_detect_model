from src.learning import train_model
import argparse

parser = argparse.ArgumentParser(description='dynamic object detection model')

parser.add_argument('--kitti_color_base_path', type=str,default="/media/jeon/hard/Kitti_dataset/dataset/sequences/")
parser.add_argument('--kitti_color_pose_path', type=str,default="/media/jeon/hard/Kitti_dataset/dataset_gray/poses/")
parser.add_argument('--kitti_color_train_seq_names', nargs='+',default=["00","05","06"], type=str)
parser.add_argument('--kitti_color_valid_seq_names', nargs='+',default=["02","08"], type=str)
parser.add_argument('--gamma', type=int,default=16)
parser.add_argument('--dilation_rate', type=int,default=4)
parser.add_argument('--learning_rate', type=float,default=0.01)
parser.add_argument('--batch_size', type=int,default=10)
parser.add_argument('--contrastive_margin', type=int,default=2)
parser.add_argument('--max_epoch', type=int,default=40)
parser.add_argument('--weight_value', type=float,default=2.0)
parser.add_argument('--bad_neighbor', type=int,default=1)
parser.add_argument('--stable_epoch', type=int, default=15)
parser.add_argument('--loss1type', type=str, default="mean")
parser.add_argument('--loss1distancetype', type=str, default="euclidean")
args = parser.parse_args()

train_model(kitti_color_base_path = args.kitti_color_base_path,
            kitti_color_pose_path = args.kitti_color_pose_path,
            train_data_names = args.kitti_color_train_seq_names,\
            validation_data_names = args.kitti_color_valid_seq_names,\
            gamma = args.gamma,\
            dilation_rate = args.dilation_rate,\
            arg_learning_rate = args.learning_rate,\
            arg_batch_size = args.batch_size,\
            arg_contrastive_margin = args.contrastive_margin,\
            max_epoch = args.max_epoch,
            weight_value = args.weight_value,
            bad_neighbor=args.bad_neighbor,
            stable_epoch=args.stable_epoch,
            loss1type=args.loss1type,
            loss1distancetype=args.loss1distancetype)