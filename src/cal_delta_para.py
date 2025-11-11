import argparse
import os
import pathlib
from copy import deepcopy
import torch
import numpy as np
from single_train_test import next_run
from alternate_train import alternate_run
wandb=None
from models import get_model
from loss import get_loss_fn
from utils import get_optimizer
from metrics import cal_llloss_with_logits, cal_auc, cal_llloss_with_logits_and_weight, cal_prauc
from data import get_criteo_dataset, DelayDataset
from tqdm import tqdm
import numpy as np
import torch
from loss import stable_log1pex
import os
import logging
import sys
from single_train_test import test
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from loss import get_loss_fn

def get_data_loader(datasets):
    train_dataset = datasets["train"]
    train_data_x = torch.from_numpy(train_dataset["x"].to_numpy().astype(np.float32))
    train_data_label = torch.from_numpy(train_dataset["labels"])
    train_data = DelayDataset(train_data_x, train_data_label)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=params["batch_size"], shuffle=True)

    valid_dataset = datasets["valid"]
    valid_data_x = torch.from_numpy(valid_dataset["x"].to_numpy().astype(np.float32))
    valid_data_label = torch.from_numpy(valid_dataset["labels"])
    valid_data = torch.utils.data.TensorDataset(valid_data_x, valid_data_label)
    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=params["batch_size"])

    test_dataset = datasets["test"]
    test_data_x = torch.from_numpy(test_dataset["x"].to_numpy().astype(np.float32))
    test_data_label = torch.from_numpy(test_dataset["labels"])
    test_data = torch.utils.data.TensorDataset(test_data_x, test_data_label)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=params["batch_size"])

    convert_batch = 1024
    one_dataset = datasets["one"]
    one_data_x = torch.from_numpy(one_dataset["x"].to_numpy().astype(np.float32))
    one_data_label = torch.from_numpy(one_dataset["labels"])
    one_data = torch.utils.data.TensorDataset(one_data_x, one_data_label)
    one_data_loader = torch.utils.data.DataLoader(one_data, batch_size=convert_batch)

    zero_dataset = datasets["zero"]
    zero_data_x = torch.from_numpy(zero_dataset["x"].to_numpy().astype(np.float32))
    zero_data_label = torch.from_numpy(zero_dataset["labels"])
    zero_data = torch.utils.data.TensorDataset(zero_data_x, zero_data_label)
    zero_data_loader = torch.utils.data.DataLoader(zero_data, batch_size=convert_batch)

    add_dataset = datasets["add"]
    add_data_x = torch.from_numpy(add_dataset["x"].to_numpy().astype(np.float32))
    add_data_label = torch.from_numpy(add_dataset["labels"])
    add_data = torch.utils.data.TensorDataset(add_data_x, add_data_label)
    add_data_loader = torch.utils.data.DataLoader(add_data, batch_size=convert_batch)

    data_loaders = {
        "train_data" : train_data_loader,
        "test_data" : test_data_loader,
        "valid_data" : valid_data_loader,
        "one_data" : one_data_loader,
        "zero_data" : zero_data_loader,
        "add_data" : add_data_loader
    }
    return train_data, data_loaders

def get_gradient(optimizer, models, data_loader):
    loss_fn_sum = get_loss_fn('vertor_loss')
    vector_gradients = []
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(data_loader)):
        batch_x = batch[0].to("cuda")
        batch_y = batch[1].to("cuda")
        targets = {"label": batch_y}
        models["model"].eval()
        outputs = models["model"](batch_x)
        loss_dict = loss_fn_sum(targets, outputs, params)
        loss = loss_dict["loss"]
        loss.backward()
    for p in models["model"].parameters():
        if p.grad is not None:
            grad_copy = ((p.grad.clone().detach())).flatten()
            vector_gradients.append(grad_copy)
    all_gradients = torch.cat(vector_gradients)
    return all_gradients

def get_vector_b(data_loaders,size_n,optimizer):
    all_gradients_zero = get_gradient(optimizer, models, data_loaders['zero_data'])
    all_gradients_one = get_gradient(optimizer, models, data_loaders['one_data'])
    vector_shape = all_gradients_zero.shape
    all_gradients =  (all_gradients_zero - all_gradients_one)/size_n
    return all_gradients,vector_shape

def get_vector_b_add(data_loaders,size_n,optimizer):
    all_gradients_zero = get_gradient(optimizer, models, data_loaders['zero_data'])
    all_gradients_one = get_gradient(optimizer, models, data_loaders['one_data'])
    all_gradients_add = get_gradient(optimizer, models, data_loaders['add_data'])
    vector_shape = all_gradients_zero.shape
    all_gradients =  (all_gradients_zero - all_gradients_one-all_gradients_add)/size_n
    return all_gradients,vector_shape

def print_args(args,logger):
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

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

def print_redirect(log_path):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename=log_path,
                    filemode='a')
    logger = logging.getLogger()
    return logger

def cal_para_change(init_value,x_lr,x_batch,x_iter,vector_b,vector_shape,optimizer,train_data,models,model_path,test_data,adjust,best_iter_gap):
    models["model"].load_state_dict(torch.load(model_path))
    initial_tensor = torch.full(vector_shape, init_value, dtype=torch.float32).cuda()
    x = torch.nn.Parameter(initial_tensor)
    optimizer_z = torch.optim.Adam([x],lr = x_lr,weight_decay = adjust)
    optimizer_z.zero_grad()
    optimizer.zero_grad()
    loss_fn = get_loss_fn('cross_entropy_loss')
    large_data_loader = torch.utils.data.DataLoader(train_data, batch_size=x_batch,shuffle=True)
    data_loader = iter(large_data_loader)

    best_auc = 0.
    best_pauc = 0.
    best_ll = 0.
    best_iter = 25

    pre_norm = torch.norm(vector_b, p=2)
    best_delta_norm = pre_norm

    for i in range(x_iter):
        try: 
            result = next(data_loader)
            batch_x = result[0].to("cuda")
            batch_y = result[1].to("cuda")
            targets = {"label": batch_y}
        except:
            data_loader = iter(large_data_loader)
            result = next(data_loader)
            batch_x = result[0].to("cuda")
            batch_y = result[1].to("cuda")
            targets = {"label": batch_y}
        models["model"].eval()
        outputs = models["model"](batch_x)
        loss_dict = loss_fn(targets, outputs, params)
        loss = loss_dict["loss"]
        optimizer.zero_grad()
        grads = torch.autograd.grad(loss, models["model"].parameters(), create_graph=True, retain_graph=True)
        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        hvp = torch.autograd.grad(flat_grads, models["model"].parameters(), grad_outputs=x, retain_graph=True)
        flat_hvps = torch.cat([p.view(-1) for p in hvp])

        x_grad = flat_hvps - vector_b
        x_grad_norm = torch.norm(x_grad, p=2)
        delat_norm = x_grad_norm - pre_norm
        pre_norm = torch.norm(x_grad, p=2)

        lossop = 0.5 * (torch.dot(x, flat_hvps)) - torch.dot(vector_b, x)
        optimizer_z.zero_grad()
        lossop.backward()
        optimizer_z.step()

        if i % 1 == 0:
            models["model"].load_state_dict(torch.load(model_path))
            original_params = [p.clone().detach() for p in models["model"].parameters()]
            vector_to_parameters(x, original_params)
            with torch.no_grad():
                for p, c in zip(models["model"].parameters(), original_params):
                    p.add_(c)
            tauc, tprauc, tllloss = test(models["model"], test_data, params)
            if i >  20:
                if abs(delat_norm) < abs(best_delta_norm):
                    best_auc = tauc
                    best_iter = i
                    best_pauc = tprauc
                    best_ll = tllloss
                    best_delta_norm = delat_norm
                    logger.info("the better")
                if (i - best_iter)> best_iter_gap:
                    logger.info("best_auc {} , best pauc{}, best ll {} , best delta norm {}, with early stop at {}".format(best_auc,best_pauc,best_ll,best_delta_norm,best_iter))
                    break
            logger.info("iter {}, auc {},prauc {}, logloss {}, delta_norm {}, norm {}".format(i,tauc,tprauc,tllloss,delat_norm,x_grad_norm))
            models["model"].load_state_dict(torch.load(model_path))
    logger.info("best_auc {}".format(best_auc))

parser = argparse.ArgumentParser()
parser.add_argument("--x_lr", type=float, default=0.0001)
parser.add_argument("--x_init_value", type=float, default=0.001)
parser.add_argument("--x_adjust", type=float, default=0)
parser.add_argument("--x_iter", type=int, default=2000)
parser.add_argument("--x_batch", type=int, default=2048)
parser.add_argument("--method", help="delayed feedback method",
                        choices=["FSIW",
                                 "DFM",
                                 "Oracle",
                                 "Vanilla",
                                 "BasicLC",
                                 "nnDF",
                                 "ULC"],
                        type=str, default="Vanilla")
parser.add_argument("--mode", type=str, choices=["pretrain", "train"], help="training mode", default="train")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset_source", type=str, default="criteo", choices=["criteo"])
parser.add_argument("--CD", type=int, default=7,
                    help="interval between counterfactual deadline and actual deadline")
parser.add_argument("--lr", type=float, default=1e-3)
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
parser.add_argument("--l2_reg", type=float, default=0,
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
parser.add_argument("--cuda_device", type=str, default="0")
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--save_model", type=int, default=0)
parser.add_argument("--base_model", type=str, default="MLP", choices=["MLP", "DeepFM", "DCNV2","AutoInt"])
parser.add_argument("--result_file", type=str,default='test_ipynb.out')
parser.add_argument("--test_gap", type=float,default=10.0)
parser.add_argument("--add", type=int,default=-1)
parser.add_argument("--best_iter_gap", type=int,default=15)

para_args = parser.parse_args()
para_args.result_file = "./iflog/"+str(para_args.base_model)+'/'+'end_'+str(para_args.training_end_day)+\
    "_duration_"+str(para_args.training_duration)+"_test_duration_"+str(para_args.test_gap)\
        +"_method_"+str(para_args.method)+ "_valid_" + str(para_args.valid_test_size)+  "_seed_" + str(para_args.seed)+"_tune.log"
if para_args.add == 1:
    para_args.result_file = "./iflog/add/"+str(para_args.base_model)+'/'+'end_'+str(para_args.training_end_day)+\
    "_duration_"+str(para_args.training_duration)+"_test_duration_"+str(para_args.test_gap)\
        +"_method_"+str(para_args.method)+ "_valid_" + str(para_args.valid_test_size)+  "_seed_" + str(para_args.seed)+"_tune_add.log"

logger = print_redirect(para_args.result_file)
print_args(para_args,logger)
params = run_params(para_args)

os.environ["CUDA_VISIBLE_DEVICES"]=params["cuda_device"]
torch.manual_seed(para_args.seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(para_args.seed)
np.random.seed(para_args.seed)

# init the model
datasets = get_criteo_dataset(params,logger)
np.random.seed(params["seed"])
model = get_model("MLP_SIG", params)
models = {"model": model.to("cuda")}
optimizer = get_optimizer(models["model"].parameters(), params["optimizer"], params)

train_data, data_loaders = get_data_loader(datasets)

model_path = "./seed_train/"+ str(params['base_model'])+'/' + \
        str(params['base_model']) + '_model_end' + str(params['training_end_day'])+"_duration_"+\
        str(params['training_duration'])+"_test_duration_"+str(params['test_gap'])+"_method_"+\
        str(params['method'])+ "_valid_" + str(params["valid_test_size"])+"_seed_" + str(params["seed"])+".pth"

models["model"].load_state_dict(torch.load(model_path))

tauc, tprauc, tllloss = test(models["model"], data_loaders['test_data'], params)
logger.info("ori_auc {} , ori_pauc {}, ori_ll {}".format(tauc,tprauc,tllloss))

if params['add'] == 1:
    b,vector_shape = get_vector_b_add(data_loaders,len(train_data),optimizer)
else:
    b,vector_shape = get_vector_b(data_loaders,len(train_data),optimizer)

cal_para_change(para_args.x_init_value,para_args.x_lr,para_args.x_batch,para_args.x_iter,b,\
vector_shape,optimizer,train_data,models,model_path,data_loaders['test_data'],para_args.x_adjust,para_args.best_iter_gap)