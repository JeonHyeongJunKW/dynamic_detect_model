from src.learning import train_model
import argparse

parser = argparse.ArgumentParser(description='dynamic object detection model')

parser.add_argument('--gamma', type=int,default=16)
parser.add_argument('--dilation_rate', type=int,default=4)
parser.add_argument('--learning_rate', type=float,default=0.01)
parser.add_argument('--batch_size', type=int,default=10)
parser.add_argument('--contrastive_margin', type=int,default=2)
parser.add_argument('--max_epoch', type=int,default=40)
parser.add_argument('--weight_value', type=float,default=1.0)
args = parser.parse_args()
data_group = ["00","05","06"]
valid_group = ["02","08"]

train_model(train_data_names = data_group,\
            validation_data_names = valid_group,\
            gamma = args.gamma,\
            dilation_rate = args.dilation_rate,\
            arg_learning_rate = args.learning_rate,\
            arg_batch_size = args.batch_size,\
            arg_contrastive_margin = args.contrastive_margin,\
            max_epoch = args.max_epoch,
            weight_value = args.weight_value)