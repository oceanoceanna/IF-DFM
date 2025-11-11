import copy
import os

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import torch
import datetime

from utils import parse_float_arg

SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)

class DelayDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple([self.tensors[0][index], self.tensors[1][index], index])

    def __len__(self):
        return self.tensors[0].size(0)

def get_data_df(params):
    if params["dataset_source"] in ["criteo"]:
        df = pd.read_csv(params["data_path"], sep="\t", header=None)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()

    if params["dataset_source"] == "criteo":
        df = df[df.columns[2:]]
        for c in df.columns[8:]:
            df[c] = df[c].fillna("")
            df[c] = df[c].astype(str)
        
        label_encoder = LabelEncoder()
        for c in df.columns[8:]:
            df[c] = label_encoder.fit_transform(df[c])

        for i, c in enumerate(df.columns[:8]):
            df[c] = df[c].fillna(-1)
            df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
            df[c] = np.floor(df[c]*(num_bin_size[i] - 0.00001)).astype(str)
        df.columns = [str(i) for i in range(17)]
    return df, click_ts, pay_ts


class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, sample_ts=None, labels=None, delay_label=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.delay_label = delay_label
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if labels is not None:
            self.labels = copy.deepcopy(labels)
        else:
            self.labels = (pay_ts > 0).astype(np.int32)

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def to_fsiw_1(self, cd, T):  # build pre-training dataset 1 of FSIW
        mask = np.logical_and(self.click_ts < T-cd, self.pay_ts > 0)
        mask = np.logical_and(mask, self.pay_ts < T)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.click_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < T - cd] = 1
        # FSIW needs elapsed time information
        x.insert(x.shape[1], column="elapse", value=(
            T-click_ts-cd)/SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_0(self, cd, T):  # build pre-training dataset 0 of FSIW
        mask = np.logical_or(self.pay_ts >= T-cd, self.pay_ts < 0)
        mask = np.logical_or(mask, self.pay_ts > T)
        mask = np.logical_and(self.click_ts < T-cd, mask)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.sample_ts[mask]
        label = np.zeros((x.shape[0],))
        label[np.logical_or(pay_ts < 0, pay_ts > T)] = 1
        x.insert(x.shape[1], column="elapse", value=(
            T-click_ts-cd)/SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.sample_ts[idx],
                      self.labels[idx])

def get_criteo_dataset(params,logger):
    name = params["dataset"]
    logger.info("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        logger.info("cache_path {}".format(cache_path))
        logger.info("\nloading data from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
        if "valid" in data:
            valid_data = data["valid"]
        if "add" in data:
            add_data = data["add"]
        if "fn" in data:
            fn_data = data["fn"]
        if "zero" in data:
            convert_data_zero = data["zero"]
        if "one" in data:
            convert_data_one = data["one"]
    else:
        print("\nbuilding dataset")

        starttime = datetime.datetime.now()
        if params["dataset_source"] == "criteo":
            source_cache_path = "./cache_data.pkl"
        if os.path.isfile(source_cache_path):
            with open(source_cache_path, "rb") as f:
                data = pickle.load(f)
        else:
            df, click_ts, pay_ts = get_data_df(params)
            data = DataDF(df, click_ts, pay_ts)
            with open(source_cache_path, "wb") as f:
                pickle.dump(data, f, protocol=4)
        endtime = datetime.datetime.now()
        print("Time:{}s".format((endtime - starttime).total_seconds()))
        if "fsiw1" in name:
            cd = parse_float_arg(name, "cd")
            logger.info("cd {}".format(cd))
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            test_data = data.sub_days(params["training_end_day"], params["training_end_day"])
            train_data = train_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=params["training_end_day"]*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=params["training_end_day"]*SECONDS_A_DAY)
        elif "fsiw0" in name:
            cd = parse_float_arg(name, "cd")
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            test_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1)
            train_data = train_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=params["training_end_day"]*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=params["training_end_day"]*SECONDS_A_DAY)
        elif "fsiwsg" in name:
            cd = parse_float_arg(name, "cd")
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            test_data = data.sub_days(params["training_end_day"]+params["test_gap"], params["training_end_day"]+params["test_gap"]+params['valid_test_size'])
            train_data = train_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=params["training_end_day"]*SECONDS_A_DAY)
            cvrs = np.reshape(train_data.pay_ts > 0, (-1, 1))
            pot_cvr = np.reshape(train_data.pay_ts > params["training_end_day"]*SECONDS_A_DAY, (-1, 1))
            train_data.labels = np.reshape(train_data.labels, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, cvrs, pot_cvr], axis=1)
            test_data = test_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=params["training_end_day"]*SECONDS_A_DAY)
        elif "Vanilla" in name:
            cd = parse_float_arg(name, "cd")
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)
            train_data.labels[mask] = 0

            train_data.x.insert(train_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - train_data.click_ts)/SECONDS_FSIW_NORM)
            fn_data =  DataDF(train_data.x.iloc[mask],
                            train_data.click_ts[mask],
                            train_data.pay_ts[mask],
                            train_data.sample_ts[mask],
                            train_data.labels[mask])
            
            # mask_convert = np.logical_and(train_data.pay_ts> (params["training_end_day"]*SECONDS_A_DAY),
            #                               train_data.pay_ts< ((params["training_end_day"]+10*params["valid_test_size"])*SECONDS_A_DAY))
            a = (train_data.pay_ts>(params["training_end_day"]*SECONDS_A_DAY)) 
            # 1 day for validation
            # 1 day before train_end_day+ test_gap_day -> conversion  data
            # that is, only edit b
            b = (train_data.pay_ts< ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY))
            mask_convert = np.logical_and(a,b)
            
            convert_data_zero =  DataDF(train_data.x.iloc[mask_convert],
                        train_data.click_ts[mask_convert],
                        train_data.pay_ts[mask_convert],
                        train_data.sample_ts[mask_convert],
                        train_data.labels[mask_convert])
            convert_data_one =  DataDF(train_data.x.iloc[mask_convert],
                        train_data.click_ts[mask_convert],
                        train_data.pay_ts[mask_convert],
                        train_data.sample_ts[mask_convert],
                        train_data.labels[mask_convert])
            convert_data_one.labels = np.ones(len(train_data.labels[mask_convert]))


            add_data_ori = data.sub_days(params["training_end_day"],params["training_end_day"]+params['test_gap']-params['valid_test_size']).shuffle()
            mask_add = add_data_ori.pay_ts > ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY)
            add_data_ori.labels[mask_add] = 0
            add_data_ori.x.insert(add_data_ori.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - add_data_ori.click_ts)/SECONDS_FSIW_NORM)

            mask_1 = add_data_ori.pay_ts > ((params["training_end_day"])*SECONDS_A_DAY)
            mask_2 = add_data_ori.pay_ts < ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY)
            mask_pos = np.logical_and(mask_1,mask_2)

            # add_data =  DataDF(add_data_ori.x.iloc[mask_pos],
            #             add_data_ori.click_ts[mask_pos],
            #             add_data_ori.pay_ts[mask_pos],
            #             add_data_ori.sample_ts[mask_pos],
            #             add_data_ori.labels[mask_pos])
            # add_data.labels = np.ones(len(add_data_ori.labels[mask_pos]))
            add_data  = add_data_ori
            # valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            # valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
            #     params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            # val_mask = valid_data.pay_ts > ((params["training_end_day"] + 1*params["valid_test_size"])*SECONDS_A_DAY)

            valid_data = data.sub_days((params["training_end_day"]+params['test_gap']-params['valid_test_size']), params["training_end_day"]+params['test_gap'])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            val_mask = valid_data.pay_ts > ((params["training_end_day"]+params['test_gap'])*SECONDS_A_DAY)
            valid_data.labels[val_mask] = 0

            test_data = data.sub_days(params["training_end_day"]+params['test_gap'], params["training_end_day"]+params['test_gap']+params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)

            logger.info("valid:{}".format(valid_data.labels.sum()))
            logger.info("train:{}".format(train_data.labels.sum()))
            logger.info("zero:{}".format(convert_data_zero.labels.sum()))
            logger.info("one:{}".format(convert_data_one.labels.sum()))
        elif "ULC" in name:
            cd = parse_float_arg(name, "cd")
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)
            train_data.labels[mask] = 0

            train_data.x.insert(train_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - train_data.click_ts)/SECONDS_FSIW_NORM)
            fn_data =  DataDF(train_data.x.iloc[mask],
                            train_data.click_ts[mask],
                            train_data.pay_ts[mask],
                            train_data.sample_ts[mask],
                            train_data.labels[mask])
            
            # mask_convert = np.logical_and(train_data.pay_ts> (params["training_end_day"]*SECONDS_A_DAY),
            #                               train_data.pay_ts< ((params["training_end_day"]+10*params["valid_test_size"])*SECONDS_A_DAY))
            a = (train_data.pay_ts>(params["training_end_day"]*SECONDS_A_DAY)) 
            # 1 day for validation
            # 1 day before train_end_day+ test_gap_day -> conversion  data
            # that is, only edit b
            b = (train_data.pay_ts< ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY))
            mask_convert = np.logical_and(a,b)
            
            convert_data_zero =  DataDF(train_data.x.iloc[mask_convert],
                        train_data.click_ts[mask_convert],
                        train_data.pay_ts[mask_convert],
                        train_data.sample_ts[mask_convert],
                        train_data.labels[mask_convert])
            convert_data_one =  DataDF(train_data.x.iloc[mask_convert],
                        train_data.click_ts[mask_convert],
                        train_data.pay_ts[mask_convert],
                        train_data.sample_ts[mask_convert],
                        train_data.labels[mask_convert])
            convert_data_one.labels = np.ones(len(train_data.labels[mask_convert]))


            add_data_ori = data.sub_days(params["training_end_day"],params["training_end_day"]+params['test_gap']-params['valid_test_size']).shuffle()
            mask_add = add_data_ori.pay_ts > ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY)
            add_data_ori.labels[mask_add] = 0
            add_data_ori.x.insert(add_data_ori.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - add_data_ori.click_ts)/SECONDS_FSIW_NORM)

            mask_1 = add_data_ori.pay_ts > ((params["training_end_day"])*SECONDS_A_DAY)
            mask_2 = add_data_ori.pay_ts < ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY)
            mask_pos = np.logical_and(mask_1,mask_2)

            add_data =  DataDF(add_data_ori.x.iloc[mask_pos],
                        add_data_ori.click_ts[mask_pos],
                        add_data_ori.pay_ts[mask_pos],
                        add_data_ori.sample_ts[mask_pos],
                        add_data_ori.labels[mask_pos])
            add_data.labels = np.ones(len(add_data_ori.labels[mask_pos]))
            # valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            # valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
            #     params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            # val_mask = valid_data.pay_ts > ((params["training_end_day"] + 1*params["valid_test_size"])*SECONDS_A_DAY)

            valid_data = data.sub_days((params["training_end_day"]+params['test_gap']-params['valid_test_size']), params["training_end_day"]+params['test_gap'])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            val_mask = valid_data.pay_ts > ((params["training_end_day"]+params['test_gap'])*SECONDS_A_DAY)
            valid_data.labels[val_mask] = 0

            test_data = data.sub_days(params["training_end_day"]+params['test_gap'], params["training_end_day"]+params['test_gap']+params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)

            logger.info("valid:{}".format(valid_data.labels.sum()))
            logger.info("train:{}".format(train_data.labels.sum()))
            logger.info("zero:{}".format(convert_data_zero.labels.sum()))
            logger.info("one:{}".format(convert_data_one.labels.sum()))
        elif "nndf_next" in name:
            cd = params["CD"]
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)
            train_data.labels[mask] = 0

            train_data.x.insert(train_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - train_data.click_ts)/SECONDS_FSIW_NORM)
            fn_data =  DataDF(train_data.x.iloc[mask],
                            train_data.click_ts[mask],
                            train_data.pay_ts[mask],
                            train_data.sample_ts[mask],
                            train_data.labels[mask])

            mask_fn = np.logical_and(train_data.pay_ts > (params["training_end_day"] - cd)*SECONDS_A_DAY, train_data.pay_ts < params["training_end_day"]*SECONDS_A_DAY)
            mask_fn = np.logical_and(train_data.click_ts < (params["training_end_day"]-cd)*SECONDS_A_DAY, mask_fn)
            cf_fn = np.reshape(mask_fn, (-1, 1))
            train_data.labels = np.reshape(train_data.labels, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, cf_fn], axis=1)

            mask_ine = train_data.click_ts < (params["training_end_day"]-cd)*SECONDS_A_DAY
            mask_ine = np.reshape(mask_ine, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, mask_ine], axis=1)

            valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            val_mask = valid_data.pay_ts > ((params["training_end_day"] + 1*params["valid_test_size"])*SECONDS_A_DAY)
            valid_data.labels[val_mask] = 0
            test_data = data.sub_days(params["training_end_day"]+1*params["valid_test_size"], params["training_end_day"]+2*params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)
        elif "dfm_next" in name:
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)
            train_data.labels[mask] = 0
            train_data.pay_ts[train_data.pay_ts < 0] = SECONDS_A_DAY*params["training_end_day"]
            delay = np.reshape(train_data.pay_ts -
                                train_data.click_ts, (-1, 1))/SECONDS_DELAY_NORM
            train_data.labels = np.reshape(train_data.labels, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, delay], axis=1)
            train_data.x.insert(train_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - train_data.click_ts)/SECONDS_FSIW_NORM)

            fn_data =  DataDF(train_data.x.iloc[mask],
                            train_data.click_ts[mask],
                            train_data.pay_ts[mask],
                            train_data.sample_ts[mask],
                            train_data.labels[mask])

            valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            val_mask = valid_data.pay_ts > ((params["training_end_day"] + 1*params["valid_test_size"])*SECONDS_A_DAY)
            valid_data.labels[val_mask] = 0
            test_data = data.sub_days(params["training_end_day"]+1*params["valid_test_size"], params["training_end_day"]+2*params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)
        elif 'retrain' in name:
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            train_data.x.insert(train_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - train_data.click_ts)/SECONDS_FSIW_NORM)

            a = (train_data.pay_ts>(params["training_end_day"]*SECONDS_A_DAY)) 
            # 1 day for validation
            # 1 day before train_end_day+ test_gap_day -> conversion  data
            # that is, only edit b
            b = (train_data.pay_ts< ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY))
            mask_convert = np.logical_and(a,b)

            mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)
            train_data.labels[mask] = 0
            train_data.labels[mask_convert] = 1
            # mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)

            fn_data =  DataDF(train_data.x.iloc[mask],
                            train_data.click_ts[mask],
                            train_data.pay_ts[mask],
                            train_data.sample_ts[mask],
                            train_data.labels[mask])
            fn_data.labels[:] = 0

            # valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            # valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
            #     params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            valid_data = data.sub_days(params["training_end_day"]+params["test_gap"]-params['valid_test_size'], params["training_end_day"]+params["test_gap"])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)

            test_data = data.sub_days(params["training_end_day"]+params["test_gap"], params["training_end_day"]+params["test_gap"]+params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)

        elif 'readd' in  name:
            training_start = params["training_end_day"] - params["training_duration"]
            train_data_ori = data.sub_days(training_start, params["training_end_day"]+params['test_gap']-params['valid_test_size']).shuffle()
            train_data_ori.x.insert(train_data_ori.x.shape[1], column="elapse", value=(
                (params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY - train_data_ori.click_ts)/SECONDS_FSIW_NORM)

            mask = train_data_ori.pay_ts > ((params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY)
            train_data_ori.labels[mask] = 0

            mask_1 = (train_data_ori.pay_ts>(params["training_end_day"])*SECONDS_A_DAY) 
            mask_2 = (train_data_ori.pay_ts<(params["training_end_day"]+params['test_gap']-params['valid_test_size'])*SECONDS_A_DAY) 
            mask_3 = (train_data_ori.click_ts>(params["training_end_day"])*SECONDS_A_DAY) 
            mask_pay = np.logical_and(mask_1,mask_2)
            mask_add = np.logical_and(mask_pay,mask_3)
            train_data_ori.labels[mask_add] = 1
            # mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)

            mask_ori_1 = (train_data_ori.click_ts<(params["training_end_day"])*SECONDS_A_DAY) 
            mask_ori_2 = (train_data_ori.click_ts>(training_start)*SECONDS_A_DAY) 
            mask_temp = np.logical_and(mask_ori_1,mask_ori_2)
            mask_readd = np.logical_or(mask_temp,mask_add)

            train_data =  DataDF(train_data_ori.x.iloc[mask_readd],
                            train_data_ori.click_ts[mask_readd],
                            train_data_ori.pay_ts[mask_readd],
                            train_data_ori.sample_ts[mask_readd],
                            train_data_ori.labels[mask_readd])
            train_data = train_data_ori
            # valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            # valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
            #     params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            valid_data = data.sub_days(params["training_end_day"]+params["test_gap"]-params['valid_test_size'], params["training_end_day"]+params["test_gap"])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)

            test_data = data.sub_days(params["training_end_day"]+params["test_gap"], params["training_end_day"]+params["test_gap"]+params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)
        elif "oracle" in name:
            cd = parse_float_arg(name, "cd")
            training_start = params["training_end_day"] - params["training_duration"]
            train_data = data.sub_days(training_start, params["training_end_day"]).shuffle()
            train_data.x.insert(train_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - train_data.click_ts)/SECONDS_FSIW_NORM)

            mask = train_data.pay_ts > (params["training_end_day"]*SECONDS_A_DAY)
            fn_data =  DataDF(train_data.x.iloc[mask],
                            train_data.click_ts[mask],
                            train_data.pay_ts[mask],
                            train_data.sample_ts[mask],
                            train_data.labels[mask])
            fn_data.labels[:] = 0

            # valid_data = data.sub_days(params["training_end_day"], params["training_end_day"]+1*params["valid_test_size"])
            # valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
            #     params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)
            valid_data = data.sub_days(params["training_end_day"]+params["test_gap"]-params['valid_test_size'], params["training_end_day"]+params["test_gap"])
            valid_data.x.insert(valid_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - valid_data.click_ts)/SECONDS_FSIW_NORM)

            test_data = data.sub_days(params["training_end_day"]+params["test_gap"], params["training_end_day"]+params["test_gap"]+params["valid_test_size"])
            test_data.x.insert(test_data.x.shape[1], column="elapse", value=(
                params["training_end_day"]*SECONDS_A_DAY - test_data.click_ts)/SECONDS_FSIW_NORM)
        if params["data_cache_path"] != "None":
            with open(cache_path, "wb") as f:
                if "Vanilla" in name:
                    pickle.dump({"train": train_data, "test": test_data, "valid": valid_data, "add":add_data,"one":convert_data_one,"zero":convert_data_zero}, f)
                    logger.info("dumping successfully!")
                elif "retrain" in name:
                    pickle.dump({"train": train_data, "test": test_data,"valid":valid_data}, f)
                    logger.info("dumping successfully!")
                elif "readd" in name:
                    pickle.dump({"train": train_data, "test": test_data, "valid": valid_data}, f)
                    logger.info("dumping successfully!")
                elif "ULC" in name:
                    pickle.dump({"train": train_data, "test": test_data, "valid": valid_data}, f)
                    logger.info("dumping successfully!")
                elif "fsiwsg" in name:
                    pickle.dump({"train": train_data, "test": test_data}, f)
                    logger.info("dumping successfully!")
    result =  {
        "train": {
            "x": train_data.x,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": train_data.labels,
        },
        "test": {
            "x": test_data.x,
            "click_ts": test_data.click_ts,
            "pay_ts": test_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": test_data.labels,
        }
    }
    if ("Vanilla" in name) or ("oracle" in name) or ('retrain' in name) or ('readd' in name) or (('ULC' in name)):
        result["valid"] = {
            "x": valid_data.x,
            "click_ts": valid_data.click_ts,
            "pay_ts": valid_data.pay_ts,
            "sample_ts": valid_data.sample_ts,
            "labels": valid_data.labels,            
        }
        if "Vanilla" in name:
            result["zero"] = {
            "x": convert_data_zero.x,
            "click_ts": convert_data_zero.click_ts,
            "pay_ts": convert_data_zero.pay_ts,
            "sample_ts": convert_data_zero.sample_ts,
            "labels": convert_data_zero.labels,            
            }     
            result["one"] = {
                "x": convert_data_one.x,
                "click_ts": convert_data_one.click_ts,
                "pay_ts": convert_data_one.pay_ts,
                "sample_ts": convert_data_one.sample_ts,
                "labels": convert_data_one.labels,            
            }    
            result["add"] = {
                "x": add_data.x,
                "click_ts": add_data.click_ts,
                "pay_ts": add_data.pay_ts,
                "sample_ts": add_data.sample_ts,
                "labels": add_data.labels,            
            }  
    return result