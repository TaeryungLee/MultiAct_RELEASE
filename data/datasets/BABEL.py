import os
import json
import copy
import cv2
import random
import pickle

import time
import torch
import numpy as np

from itertools import product

from data.datautils.transforms import *
from data.datautils.babel_label import label as BABEL_label
from data.datautils.babel_label import label_over_ten
from data.datautils.babel_label import label_over_twenty 


BABEL_label_rev = {v:k for k, v in BABEL_label.items()}



class BABEL(torch.utils.data.Dataset):
    def __init__(self, cfg, split="train", sampled_file_path=None, sampled=None, filter_label=None, cap=None):
        super().__init__()

        print("loading {} data...".format(split))
        start_time = time.time()

        self.split = split

        self.data_dir = os.path.join(cfg.data_dir, "BABEL")
        self.annot_data_dir = os.path.join(self.data_dir, "babel_v1.0_release")
        self.motion_data_dir = os.path.join(self.data_dir, "AMASS")

        self.motion_data = []
        self.metadata = []
        self.annotation = []

        self.sampled = []
        self.all_sampled = []
        self.cfg = cfg

        self.sample_fps = cfg.sample_fps
        self.maxlen = cfg.max_input_len
        self.minlen = cfg.min_input_len

        self.extra_S2_frames = cfg.extra_S2_frames[split]

        if filter_label is not None:
            self.filter_label = filter_label
        else:
            self.filter_label = cfg.filter_label

        if cap is not None:
            self.filter_cap = cap
        else:
            self.filter_cap = 'none'

        self.load_data(split)
        print("data loading done in ", round(time.time() - start_time, 2), " s")
        print("using {} raw data".format(len(self.metadata)))
        print("now sampling data")

        if sampled_file_path is not None:
            with open(sampled_file_path, "rb") as f:
                self.sampled = pickle.load(f)
            print("loaded dataset from given sampled file")
            print("using {} sampled {} pairs\n".format(len(self.sampled), split))
        elif sampled is not None:
            self.sampled = sampled
            print("loaded dataset from given sampled object")
            print("using {} sampled {} pairs\n".format(len(self.sampled), split))

        else:
            if split ==  "train":
                start_time = time.time()
                self.sample_train_data(cfg.sampling)
                print("data sampling done in ", round(time.time() - start_time, 2), " s")
                print("using {} sampled training pairs\n".format(len(self.sampled)))
            
            elif split == "val":
                start_time = time.time()
                self.sample_train_data(cfg.sampling)
                print("data sampling done in ", round(time.time() - start_time, 2), " s")
                print("using {} sampled validation pairs\n".format(len(self.sampled)))

    def dump_sampled(self, name):
        sampled_pair = [sample["labels"] for sample in self.sampled]
        with open("preprocess/sampled_pairs_{}.pkl".format(name), "wb") as f:
            pickle.dump(sampled_pair, f)


    def __len__(self):
        return len(self.sampled)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.sampled[idx])
        reg_param = self.motion_data[data["idx"]]["poses"]
        reg_param = reg_param[data["frame_filter"]]

        reg_trans = self.motion_data[data["idx"]]["trans"]
        reg_trans = reg_trans[data["frame_filter"]]

        reg_param, reg_trans = face_front_align(reg_param, reg_trans, 
            reg_param[data["frame_length"][0]-1, :3], reg_trans[data["frame_length"][0]-1, :3]
        )

        dummy_len = self.maxlen - reg_param.shape[0]
        reg_param = torch.cat((reg_param, torch.zeros((dummy_len, reg_param.shape[1])))).to(torch.float32)
        reg_trans = torch.cat((reg_trans, torch.zeros((dummy_len, reg_trans.shape[1])))).to(torch.float32)

        """
        Input data
        
        idx, smpl_trans, smpl_param: trivial
        labels: tensor(3)
            [action label of S1, "transition", action label of S2]
        label_mask: tensor(max_length_of_motion)
            [label corresponding of each frame]
        frame_length: tensor(3)
            [length of S1, length of T, length of S2]
        valid_mask: tensor(max_length_of_motion)
            1 for valid frames, 0 for invalid frames
            [1] * motion_length + [0] * (max_length_of_motion - motion_length)
        subaction_mean_mask: tensor(max_length_of_motion)
            [1/(length of S1)] * length of S1 + [1/(length of T)] * length of T + [1/(length of S2)] * length of S2 + [0] * (max_length_of_motion - motion_length)
        subaction_mask: tensor(max_length_of_motion)
            [0] * length of S1 + [1] * length of T + [2] * length of S2 + [-1] * (max_length_of_motion - motion_length)
        output_mask: tensor(max_length_of_motion)
            [1] * length of T + [2] * length of S2 + [0] * (remaining part)
        standalone_mask: tensor(max_length_of_motion)
            [1] * length_of_S2 + [0] * (remaining part)
        """

        inputs = {
            "idx": torch.tensor((data["idx"])),
            "smpl_trans": reg_trans,
            "smpl_param": reg_param,
            "S1_end_mask": data["S1_end_mask"],
            "label_mask": torch.tensor(data["label_mask"], dtype=torch.int64),
            "labels": torch.tensor(data["labels"], dtype=torch.int64),
            "valid_mask": torch.tensor(data["valid_mask"]),
            "subaction_mean_mask": torch.tensor(data["subaction_mean_mask"], dtype=torch.float32),
            "frame_length": data["frame_length"],
            "subaction_mask": torch.tensor(data["subaction_mask"], dtype=torch.int64),
            "output_mask": torch.tensor(data["output_mask"], dtype=torch.int64),
            "standalone_mask": torch.tensor(data["standalone_mask"], dtype=torch.int64),
        }

        return inputs


    def filter_train_data(self, cut="twenty"):
        assert cut in ["twenty", "ten"]
        cap = self.filter_cap
        if cut == "twenty":
            label_list = [BABEL_label[l] for l in label_over_twenty]

        elif cut == "ten":
            label_list = [BABEL_label[l] for l in label_over_ten]
        
        count = {l:0 for l in label_list}

        filtered = []

        random.shuffle(self.all_sampled)

        if cap == 'none' or self.cfg.dev_mode == True:
            cap = 1000
            for sample in self.all_sampled:
                    a2 = int(sample['labels'][-1])
                    if a2 not in label_list:
                        continue
                    if count[a2] >= cap:
                        continue
                    else:
                        count[a2] += 1
                        filtered.append(sample)
        else:
            while sum([v for _, v in count.items()]) < cap * len(label_list):
                for sample in self.all_sampled:
                    a2 = int(sample['labels'][-1])
                    if a2 not in label_list:
                        continue
                    if count[a2] >= cap:
                        continue
                    else:
                        count[a2] += 1
                        filtered.append(sample)

        self.sampled = filtered


    
    def sample_train_data(self, method='basic'):
        # idx, frame filter, cat, label
        before_sampling = []
        for idx, (motion, metadata, annotation) in enumerate(zip(self.motion_data, self.metadata, self.annotation)):
            fps = metadata["fps"]
            sample_rate = int(fps / self.sample_fps)
            
            # annotation processing
            # adjacent two: (S1 Tr S2)
            if method == "adjacent_two":
                for S1, S2 in product(annotation, annotation):
                    # adjacnecy
                    if S1["end_frame"] != S2["start_frame"]:
                        continue
                    
                    # without transition
                    if S1["label"] == "transition" or S2["label"] == "transition":
                        continue
                    
                    # add transition manually
                    S1_len = S1["end_frame"] - S1["start_frame"]
                    S2_len = S2["end_frame"] - S2["start_frame"]
                    transition_rate = self.cfg.transition_rate * 2
                    max_transition_len = self.cfg.max_transition_len * sample_rate * 2
                    S1_transition_len = int(max(min(S1_len * transition_rate, max_transition_len), 1))
                    S2_transition_len = int(max(min(S2_len * transition_rate, max_transition_len), 1))

                    # subsequence info: 
                    # labels: labels in english
                    # startframe: starting frame of each subsequence before sampling
                    # len_series: length of each subsequence before sampling
                    labels = [S1["label"], "transition", S2["label"]]
                    startframes = [S1["start_frame"], S2["start_frame"] - S1_transition_len, S2["start_frame"] + S2_transition_len]
                    len_series = [S1_len - S1_transition_len, S1_transition_len + S2_transition_len, S2_len - S2_transition_len]

                    if abs(metadata["num_frames"] - (S2["end_frame"])) >= 3 and sum(len_series) < sample_rate * (self.maxlen - self.extra_S2_frames - 1):
                        len_series[2] += self.extra_S2_frames * sample_rate
                    
                    # condition: maximum sequence length
                    length = sum(len_series)
                    if length >= sample_rate * (self.maxlen - 1):
                        continue

                    before_sampling.append((idx, labels, startframes, len_series, sample_rate, (S1_transition_len/sample_rate, S2_transition_len/sample_rate)))

                # 2.2. (S1 Tr S2): with transition
                for S1, Tr, S2 in product(annotation, annotation, annotation):
                    # adjacency
                    if S1["end_frame"] != Tr["start_frame"] or Tr["end_frame"] != S2["start_frame"]:
                        continue

                    # without transition
                    if S1["label"] == "transition" or Tr["label"] != "transition" or S2["label"] == "transition":
                        continue
                    
                    S1_len = S1["end_frame"] - S1["start_frame"]
                    S2_len = S2["end_frame"] - S2["start_frame"]
                    S1_transition_len = 0
                    S2_transition_len = 0

                    transition_rate = self.cfg.transition_rate
                    max_transition_len = self.cfg.max_transition_len * sample_rate
                    S1_transition_len = int(max(min(S1_len * transition_rate, max_transition_len), 1))
                    S2_transition_len = int(max(min(S1_len * transition_rate, max_transition_len), 1))

                    labels = [S1["label"], Tr["label"], S2["label"]]
                    startframes = [S1["start_frame"], Tr["start_frame"]-S1_transition_len, S2["start_frame"]+S2_transition_len]

                    len_series = [
                        S1["end_frame"] - S1["start_frame"] - S1_transition_len, 
                        Tr["end_frame"] - Tr["start_frame"] + S1_transition_len + S2_transition_len, 
                        S2["end_frame"] - S2["start_frame"] - S2_transition_len]

                    if abs(metadata["num_frames"] - (S2["end_frame"])) >= 3 and sum(len_series) < sample_rate * (self.maxlen - self.extra_S2_frames - 1):
                        len_series[2] += self.extra_S2_frames * sample_rate

                    # condition: maximum sequence length
                    length = sum(len_series)
                    if length >= sample_rate * (self.maxlen - 1):
                        continue

                    before_sampling.append((idx, labels, startframes, len_series, sample_rate, (S1_transition_len/sample_rate, S2_transition_len/sample_rate)))
            
            else:
                raise NotImplementedError("Unknown data sampling method")

        # sampling
        for (idx, labels, startframes, len_series, sample_rate, (S1_transition_len, S2_transition_len)) in before_sampling:
            frame_filter = np.arange(
                start=startframes[0], 
                stop=min(startframes[0]+sum(len_series), self.motion_data[idx]["poses"].shape[0]-1), 
                step=sample_rate)
            start_frame = [int(np.sum(len_series[:k])) for k in range(len(len_series))]
            start_frame = [fr // sample_rate for fr in start_frame]
            start_frame = [start_frame[i] + int(i != 0)  for i in range(len(start_frame))]
            framelength = [start_frame[i+1] - start_frame[i] for i in range(len(start_frame)-1)]
            framelength.append(int(len(frame_filter) - np.sum(framelength)))
            # valid mask, subaction_mask, subaction_square_mask, subaction_startframe
            valid_mask = np.zeros((self.maxlen, ))
            valid_mask[:int(np.sum(framelength))] = 1
            
            subaction_mask = np.ones((self.maxlen, )) * (-1)
            output_mask = np.zeros((self.maxlen, ))
            standalone_mask = np.zeros((self.maxlen, ))
            subaction_mean_mask = np.zeros((self.maxlen, 3))
            action_label = np.zeros((self.maxlen, ))

            invalid_label = False
            for label in labels:
                if label not in BABEL_label.keys():
                    invalid_label = True

            if invalid_label:
                continue
            
            if (framelength[0] < 3) or (framelength[1] < 3) or (framelength[2] < 3):
                continue

            if sum(framelength) >= self.maxlen:
                continue
            
            # S1_end_mask
            S1_end_len = self.cfg.S1_end_len
            if framelength[0] < S1_end_len + 1:
                continue

            cur = 0
            for maskno, (l, label) in enumerate(zip(framelength, labels)):
                for p in range(cur, cur+l):
                    subaction_mask[p] = maskno
                    action_label[p] = BABEL_label[label]
                    subaction_mean_mask[p, maskno] = 1./l
                cur += l
            
            cur = 0
            for maskno, (l, label) in enumerate(zip(framelength[1:], labels[1:])):
                for p in range(cur, cur+l):
                    output_mask[p] = maskno + 1
                cur += l
            
            cur = 0
            for maskno, (l, label) in enumerate(zip(framelength[2:], labels[2:])):
                for p in range(cur, cur+l):
                    standalone_mask[p] = maskno + 1
                cur += l

            num_labels = [BABEL_label[label] for label in labels]
            
            S1_end_mask = [0] * (framelength[0] - S1_end_len) + [1] * S1_end_len + [0] * (self.maxlen - framelength[0])

            assert(len(S1_end_mask) == self.maxlen)

            sampled = {
                "idx": idx,
                "frame_filter": frame_filter,
                "label_mask": action_label,
                "labels": num_labels,
                "valid_mask": valid_mask,
                "subaction_mean_mask": subaction_mean_mask,
                "frame_length": torch.tensor(framelength),
                "S1_end_mask": torch.tensor(S1_end_mask),
                "subaction_mask": subaction_mask,
                "output_mask": output_mask,
                "standalone_mask": standalone_mask,
                "add_transition_len": [S1_transition_len, S2_transition_len],
            }

            if frame_filter.shape[0] <= self.minlen:
                continue

            self.all_sampled.append(sampled)
        
        self.filter_train_data(self.filter_label)
        
        av = np.mean([f["frame_filter"].shape[0] for f in self.sampled])
        
        frame_length_mean = torch.mean(torch.cat([f["frame_length"].unsqueeze(0) for f in self.all_sampled], dim=0).to(torch.float32), dim=0).tolist()
        add_transition_mean = torch.mean(torch.tensor([f["add_transition_len"] for f in self.all_sampled]).to(torch.float32), dim=0).tolist()

        print("")
        print("all sampled: ", len(self.all_sampled))
        print("average length: ", round(av, 2))
        print("average subsequence length: ", frame_length_mean)
        print("average added trans length: ", add_transition_mean)
        print("")


    def load_data(self, split):
        with open(os.path.join(self.annot_data_dir, split + ".json"), "r") as f:
            annot = json.load(f)
        
        if self.cfg.dev_mode:
            annot = {k:annot[k] for k in list(annot.keys())[:200]}
        for idx, ann in annot.items():
            npfile = os.path.join(self.motion_data_dir, ann["feat_p"])
            motion_npz = np.load(npfile)

            # raw motion data
            motion_data = {
                "trans": motion_npz["trans"],
                "poses": motion_npz["poses"]
            }

            # meta info
            metadata = {
                "raw_idx": idx,
                "fps": round(float(motion_npz["mocap_framerate"])),
                "num_frames": motion_npz["poses"].shape[0],
                "duration": ann["dur"]
            }

            # action categories / labels
            if ann["frame_ann"] is None:
                # single label for entire motion sequence
                annotation = [
                    {
                        "label": seq_label["proc_label"],
                        "category": seq_label["act_cat"],
                        "start_frame": 0,
                        "end_frame": motion_npz["poses"].shape[0] - 1
                    }
                    for seq_label in ann["seq_ann"]["labels"]
                ]
            else:
                annotation = [
                    {
                        "label": frame_label["proc_label"],
                        "start_frame": int(frame_label["start_t"] * motion_npz["mocap_framerate"]),
                        "end_frame": int(frame_label["end_t"] * motion_npz["mocap_framerate"]),
                        "category": frame_label["act_cat"]
                    }
                    for frame_label in ann["frame_ann"]["labels"]
                ]
                annotation = sorted(annotation, key= lambda x : x["start_frame"])

            same = []
            for i in range(len(annotation)):
                if annotation[i]["start_frame"] == annotation[i]["end_frame"]:
                    same.append(i)
            for i in same:
                del annotation[i]

            self.motion_data.append(motion_data)
            self.metadata.append(metadata)
            self.annotation.append(annotation)

