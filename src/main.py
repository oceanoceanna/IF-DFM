import argparse
import os
import pathlib
from copy import deepcopy

import torch
import numpy as np

from pretrain import run
from single_train_test import next_run
from alternate_train import alternate_run
#import wandb
wandb=None

import os
import logging
import sys

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'

def run_params(args):
    params = deepcopy(vars(args))
    params["model"] = "MLP_SIG"
    if args.data_cache_path != "None":
        pathlib.Path(args.data_cache_path).mkdir(parents=True, exist_ok=True)
    if args.mode == "pretrain":
        if args.method == "FSIW":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = args.fsiw_pretraining_type+"_cd_"+str(args.CD)+"_end_"+str(args.training_end_day)+"_seed_"+str(args.seed) 
            params["model"] = "MLP_FSIW"
        else:
            raise ValueError(
                "{} method do not need pretraining other than Pretrain".format(args.method))
    else:
        if args.method == "Oracle":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == 'retrain':
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "FSIW":
            params["loss"] = "fsiw_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "BasicLC":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "nnDF":
            params["loss"] = "non_negative_loss"
            params["dataset"] = "_cd_"+str(args.CD)+"_end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "ULC":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed) +"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "Vanilla":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed)+"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        elif args.method == "readd":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed)+"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)
        else:
            params["dataset"] = "end_"+str(args.training_end_day)+"_seed_"+str(args.seed)+"_test_gap_"+str(args.test_gap) + "_duration_"+ str(args.training_duration) + "_valid_" + str(args.valid_test_size) + "_method_" + str(args.method)

    return params

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="delayed feedback method",
                        choices=["FSIW",
                                 "DFM",
                                 "Oracle",
                                 "Vanilla",
                                 "BasicLC",
                                 "nnDF",
                                 "ULC",'retrain','readd'],
                        type=str, default='Vanilla')
    parser.add_argument(
        "--mode", type=str, choices=["pretrain", "train"], default='train')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_source", type=str, default="criteo", choices=["criteo"])
    parser.add_argument("--CD", type=int, default=7,
                        help="interval between counterfactual deadline and actual deadline")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--data_path", type=str,default="./data/data.txt",
                        help="path of the data.txt in criteo dataset")
    parser.add_argument("--data_cache_path", type=str, default="./data")
    parser.add_argument("--model_ckpt_path", type=str,
                        help="path to save pretrained model")
    parser.add_argument("--pretrain_fsiw0_model_ckpt_path", type=str, default= "./models/fsiw0",
                        help="path to the checkpoint of pretrained fsiw0 model")
    parser.add_argument("--pretrain_fsiw1_model_ckpt_path", type=str, default= "./models/fsiw1",
                        help="path to the checkpoint of pretrained fsiw1 model")
    parser.add_argument("--fsiw_pretraining_type", choices=["fsiw0", "fsiw1"], type=str, default="None",
                        help="FSIW needs two pretrained weighting model")
    parser.add_argument("--batch_size", type=int,
                        default=1024)
    parser.add_argument("--epoch", type=int, default=5,
                        help="training epoch of pretraining")
    parser.add_argument("--l2_reg", type=float, default=0.0001,
                        help="l2 regularizer strength")
    parser.add_argument("--training_end_day", type=int, default=15,
                        help="deadline for training data")
    parser.add_argument("--training_duration", type=int, default=14,
                        help="duration of training data")
    parser.add_argument("--valid_test_size", type=float, default=1.0,
                        help="duration of valid/test data")
    parser.add_argument("--train_epoch", type=int, default=100,
                        help="max train epoch")
    parser.add_argument("--early_stop", type=int, default=4)
    parser.add_argument("--cuda_device", type=str, default="5")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--save_model", type=int, default=0)
    parser.add_argument("--base_model", type=str, default="MLP", choices=["MLP", "DeepFM", "DCNV2","AutoInt"])
    parser.add_argument("--result_file", type=str,default='output.log')
    parser.add_argument("--test_gap", type=float,default=10.0)
    
    args = parser.parse_args()
    args.result_file = "./trained_log_seed/" + str(args.base_model) + "/_end_"+str(args.training_end_day)+\
    "_duration_"+str(args.training_duration)+"_test_duration_"+str(args.test_gap)\
        +"_method_"+str(args.method)+ "_valid_" + str(args.valid_test_size)+ "_seed_" + str(args.seed) + "_train.log"
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=args.result_file,
                        filemode='a')
    logger = logging.getLogger()
    def print_to_log(*args, **kwargs):
        logger.info(' '.join(map(str, args)))
    print = print_to_log

    print_args(args)
    
    params = run_params(args)
    os.environ["CUDA_VISIBLE_DEVICES"]=params["cuda_device"]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "pretrain":
        run(params)
    else:
        #wandb.init(project="ULC", entity="xxxx", config = params)
        if params["method"] == "ULC":
            alternate_run(params, wandb,logger)
        else:
            next_run(params, wandb,logger)