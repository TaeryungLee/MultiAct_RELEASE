import os
import torch
import numpy as np
import imageio
import pickle

from torch.nn import DataParallel

from data.dataset import get_dataloader
from data.datautils.babel_label import label as BABEL_label
from data.datautils.transforms import *
from models.model import get_model

from utils.dir import mkdir
from utils.vis import *
from utils.human_models import SMPLH

BABEL_label_rev = {v:k for k, v in BABEL_label.items()}

class Generator():
    def __init__(self, cfg, mode, model=None):
        self.cfg = cfg
        self.smplh = SMPLH(cfg)
        self.cfg.batch_size = self.cfg.gen_batch_size

        self.align_method = face_front_align

        if model is not None:
            self.model = model
        else:
            print("model loaded from ", cfg.test_model_path)
            self.model = DataParallel(get_model(cfg)).cuda()
            self.load_model(cfg.test_model_path)
        self.model.eval()
            
        self.val_loader = get_dataloader(self.cfg, "val", drop_last=False, cap=cfg.test_per_label)

        # generation setting
        self.gen_mode = mode

        # path setting
        self.gen_short_out_dir = os.path.join(cfg.vis_dir, "single_step_unseen")
        self.gen_long_out_dir = os.path.join(cfg.vis_dir, "long_term")

        mkdir(self.gen_short_out_dir)
        mkdir(self.gen_long_out_dir)

        if self.gen_mode == "gen_long":
            self.target_labels = cfg.target_labels
            self.resume = cfg.resume
            # directory goes by: vis/long_term/exp/sample/(step_by_step motion)
            self.resume_exp = cfg.resume_exp
            self.resume_sample = cfg.resume_sample
            self.resume_step = cfg.resume_step

            # gen dir
            existing_dirs = os.listdir(self.gen_long_out_dir)
            dirnames = [str(i).zfill(3) for i in range(1000)]
            dirname = ""
            for dirn in dirnames:
                if dirn not in existing_dirs:
                    dirname = dirn
                    break
            
            self.current_exp = dirname
            dirname = os.path.join(self.gen_long_out_dir, dirname)
            os.makedirs(dirname)
            for i in range(self.cfg.gen_batch_size):
                os.makedirs(os.path.join(dirname, str(i).zfill(2)))

            self.gen_long_out_dir = dirname
        
        elif self.gen_mode == "gen_short":
            pass

        else:
            raise NotImplementedError("unknown generation mode")

        
        # visualization settings
        self.vis_step = False
        self.vis_long = True
        self.vis_long_3view = False

        self.all_front = False
        self.right_view = False
        self.fix_center = True

    
    def generate_short_term(self):
        print("generating single-step motions")
        with torch.no_grad():
            for itr, inputs in enumerate(self.val_loader):
                _, out = self.model(inputs, "gen")
                labels = inputs["labels"][:, 2].tolist()
                text_label = [BABEL_label_rev[l] for l in labels]
                valid_mask = inputs["valid_mask"]
                label_texts = []
                frame_lengths = inputs["frame_length"]
                
                for i in range(len(text_label)):
                    # S1: Input: Unseen previous motion
                    # T: Generated transition
                    # S2: Generated action motion < ~~~ >
                    frame_length = frame_lengths[i]
                    lab = text_label[i]
                    S1_text = "Input: Unseen previous motion from test set"
                    T_text = "Output: Transition"
                    S2_text = "Output: Action motion of label \"" + lab + "\""
                    
                    label_text = [S1_text] * frame_length[0] + [T_text] * frame_length[1] + [S2_text] * frame_length[2]
                    label_texts.append(label_text)

                gen_mesh = out["gen_smpl_mesh"].detach().cpu()
                names = range(1, self.cfg.gen_batch_size + 1)
                names = [str(name + itr*self.cfg.gen_batch_size).zfill(3) for name in names]

                for i in range(len(text_label)):
                    names[i] = names[i]+"_"+text_label[i]

                gen_paths = [self.gen_short_out_dir for name in names]
                visualize_batch(self.cfg, gen_mesh, valid_mask, gen_paths, names, label_texts, method="render")


    def generate_long_term(self):
        print("generating long-term motions")

        # print: exp_name, resume, resume_exp, resume_step, target_labels
        print("exp_name: ", self.current_exp)
        print("resume: ", self.resume)
        print("resume_exp: ", self.resume_exp)
        print("resume_step: ", self.resume_step)
        print("target_labels: ", self.target_labels)

        target_num_labels = [BABEL_label[lab] for lab in self.target_labels]
        results = {k:[] for k in range(len(self.target_labels))}

        # stats of motion length in training and validation set
        # it is used to determine the length of generated motion
        with open("./recursive/stats.pkl", "rb") as f:
            stats = pickle.load(f)
        
        # formulate the initial batched input
        resume_step = self.resume_step
        if self.resume:
            with open(os.path.join(self.gen_long_out_dir, str(self.resume_exp).zfill(3), "results.pkl"), "rb") as f:
                loaded_results = pickle.load(f)
                for i in range(resume_step):
                    for j in range(self.cfg.gen_batch_size):
                        results[i].append(loaded_results[i][self.resume_sample])
        else:
            resume_step = 1
            for _, inputs in enumerate(self.val_loader):
                with torch.no_grad():
                    _, out = self.model(inputs, "gen")
                
                # batchify the output
                batch_results = self.form_into_result(inputs, out, 0)

                for result in batch_results:
                    if result['label'] == target_num_labels[0]:
                        results[0].append(result)
                
                if len(results[0]) > self.cfg.gen_batch_size:
                    break
        del self.val_loader


        # recurrently generate next step
        for recur in range(resume_step, len(self.target_labels)):
            print("recur: ", recur, "\n")

            # action labels
            label = target_num_labels[recur]
            prev_label = target_num_labels[recur-1]

            # determine length of generated motion from statistics
            label_stat = stats[label]
            tr_len = int(sum(label_stat["tr_len"])/len(label_stat["tr_len"]))
            S2_len = int(sum(label_stat["S2_len"])/len(label_stat["S2_len"])) + 2
            

            # prepare input data from previous results
            samples = []
            for prev_sample in results[recur-1]:
                frame_length = [int(prev_sample['S2_len']-2), tr_len, S2_len]
                if sum(frame_length) > 115:
                    frame_length[-1] = 115 - tr_len - (int(prev_sample['S2_len']))
                prev_smpl_trans = prev_sample['gen_S2_smpl_trans'].clone()
                prev_smpl_param = prev_sample['gen_S2_smpl_param'].clone()
                smpl_param, smpl_trans = self.align_method(prev_smpl_param, prev_smpl_trans,
                    prev_smpl_param[frame_length[0]-1, :3].clone(), prev_smpl_trans[frame_length[0]-1, :3].clone()
                )

                # build inputs
                labels = [prev_label, 4, label]
                label_mask = torch.zeros((self.cfg.max_input_len, )).to(torch.int64)
                subaction_mask = torch.zeros((self.cfg.max_input_len, )).to(torch.int64)
                output_mask = torch.zeros((self.cfg.max_input_len, )).to(torch.int64)
                standalone_mask = torch.zeros((self.cfg.max_input_len, )).to(torch.int64)
                valid_mask = torch.zeros((self.cfg.max_input_len, ))
                valid_mask[:int(sum(frame_length))] = 1

                cur = 0
                for maskno, (l, label) in enumerate(zip(frame_length, labels)):
                    for p in range(cur, cur+l):
                        subaction_mask[p] = maskno
                        label_mask[p] = label
                    cur += l
                
                cur = 0
                for maskno, (l, label) in enumerate(zip(frame_length[1:], labels[1:])):
                    for p in range(cur, cur+l):
                        output_mask[p] = maskno + 1
                    cur += l
                
                cur = 0
                for maskno, (l, label) in enumerate(zip(frame_length[2:], labels[2:])):
                    for p in range(cur, cur+l):
                        standalone_mask[p] = maskno + 1
                    cur += l

                S1_end_mask = [0] * (frame_length[0] - self.cfg.S1_end_len) + [1] * self.cfg.S1_end_len + [0] * (self.cfg.max_input_len - frame_length[0])
                smpl_param[subaction_mask!=0] = 0
                smpl_trans[subaction_mask!=0] = 0
                smpl_param[valid_mask==0] = 0
                smpl_trans[valid_mask==0] = 0

                inp = {
                    'smpl_param': smpl_param,
                    'smpl_trans': smpl_trans,
                    'label_mask': label_mask,
                    'labels': torch.tensor(labels),
                    'frame_length': torch.tensor(frame_length),
                    'S1_end_mask': torch.tensor(S1_end_mask),
                    'valid_mask': valid_mask,
                    'subaction_mask': subaction_mask,
                    'output_mask': output_mask,
                    'standalone_mask': standalone_mask,
                }
                samples.append(inp)


            # batchify and load on device
            device = torch.device("cuda:0")
            batch_size = self.cfg.gen_batch_size

            num_batch = len(samples) // batch_size
            if len(samples) % batch_size != 0:
                num_batch += 1
            
            for itr in range(num_batch):
                is_last_batch = itr == (num_batch - 1)
                inputs = {}
                if not is_last_batch:
                    itr_items = samples[itr*batch_size:(itr+1)*batch_size]
                else:
                    itr_items = samples[itr*batch_size:]
                keys = list(itr_items[0].keys())
                for key in keys:
                    inputs[key] = []
                for item in itr_items:
                    for key in keys:
                        inputs[key].append(item[key].unsqueeze(0))
                for key in keys:
                    inputs[key] = torch.cat(inputs[key], dim=0).to(device)

                with torch.no_grad():
                    _, out = self.model(inputs, "gen")

                batch_results = self.form_into_result(inputs, out, recur)

                for result in batch_results:
                    results[recur].append(result)

        del self.model

        # dump result
        with open(os.path.join(self.gen_long_out_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f)
        print("exp dir name: ", self.gen_long_out_dir)


        # visualize each step
        if self.vis_step:
            for sampleno in range(self.cfg.gen_batch_size):
                gen_path = os.path.join(self.gen_long_out_dir, str(sampleno).zfill(2))
                for recur in range(resume_step, len(self.target_labels)):
                    result = results[recur][sampleno]
                    mesh = result["gen_all_mesh"].unsqueeze(0)
                    all_mask = result["all_mask"].unsqueeze(0)
                    label_mask = result["label_mask"].unsqueeze(0)
                    if recur != 0:
                        prev_label = self.target_labels[recur-1]
                    else:
                        prev_label = "random"
                    cur_label = self.target_labels[recur]
                    label_texts = [[BABEL_label_rev[int(label_idx)] for label_idx in seq] for seq in label_mask]
                    visualize_batch(self.cfg, mesh, all_mask, [gen_path], [str(recur)], label_texts, out_size=400, method="render")



        # align (=uncanonicalize)
        if self.vis_long:
            smplh = SMPLH(self.cfg)
            layer = smplh.layer
            names = []
            for sampleno in range(self.cfg.gen_batch_size):
                gen_path = os.path.join(self.gen_long_out_dir, str(sampleno).zfill(2))
                first = results[0][sampleno]
                first_framelength = first["frame_length"]
                first_param = first["gen_all_param"]
                first_trans = first["gen_all_trans"]
                first_alllen = first["all_len"]

                # cut off the initial motion, because it is taken from GT dataset
                # as a result, visualized long-term motion is entirely generated from our MultiAct
                unaligned_prev_param = first_param[first_framelength[0]:first_alllen-2]
                unaligned_prev_trans = first_trans[first_framelength[0]:first_alllen-2]
                
                for i in range(unaligned_prev_param.shape[0]):
                    names.append(self.target_labels[0])
                
                
                for recur in range(1, len(self.target_labels)):
                    # align next motion to start from the last frame of previous motion
                    target_prev_param = unaligned_prev_param[-1, :3]
                    target_prev_trans = unaligned_prev_trans[-1, :3]
                    
                    next_ = results[recur][sampleno]
                    next_framelength = next_["frame_length"]
                    next_param = next_["gen_all_param"]
                    next_trans = next_["gen_all_trans"]
                    next_alllen = next_["all_len"]
                    
                    for i in range(next_framelength[1]):
                        names.append("transition")

                    for i in range(next_alllen - next_framelength[0] - next_framelength[1] - 2):
                        names.append(BABEL_label_rev[next_['label']])

                    unaligned_next_param = next_param[next_framelength[0]:next_alllen-2]
                    unaligned_next_trans = next_trans[next_framelength[0]:next_alllen-2]

                    target_next_param = next_param[next_framelength[0]-1, :3]
                    target_next_trans = next_trans[next_framelength[0]-1, :3]

                    aligned_prev_param, aligned_prev_trans = self.align_method(
                        unaligned_prev_param, unaligned_prev_trans, target_prev_param, target_prev_trans
                    )

                    aligned_next_param, aligned_next_trans = self.align_method(
                        unaligned_next_param, unaligned_next_trans, target_next_param, target_next_trans
                    )

                    # append into single motion
                    unaligned_prev_param = torch.cat([aligned_prev_param, aligned_next_param], dim=0)
                    unaligned_prev_trans = torch.cat([aligned_prev_trans, aligned_next_trans], dim=0)
                
                # finally, bring the initial point of the motion back to origin
                reg_param, reg_trans = self.align_method(
                    unaligned_prev_param, unaligned_prev_trans, unaligned_prev_param[0, :3], unaligned_prev_trans[0, :3]
                )

                if self.all_front:
                    reg_param = to_front_view(reg_param)
                if self.right_view:
                    reg_param = to_right_view(reg_param)
                if self.fix_center:
                    reg_trans[:, :2] = 0

                # generate mesh and visualize
                reg_mesh = layer(
                    pose_body=reg_param[:, 3:66],
                    pose_hand=reg_param[:, 66:156],
                    root_orient=reg_param[:, :3],
                    trans=reg_trans
                ).v

                num_frames = reg_param.shape[0]
                frames = []
                reg_vis = vis_motion(reg_mesh, smplh.face, torch.ones((num_frames)))
                speed_sec = {'duration': 0.05}
                for i in range(num_frames):
                    img = reg_vis[i, :, :, :].astype(np.uint8)
                    cv2.putText(img, names[i], (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    frames.append(img)
                    
                imageio.mimsave(gen_path + "/long.gif", frames, **speed_sec)
                torch.cuda.empty_cache()
                del reg_mesh
                del reg_vis
                del frames

        return


    def form_into_result(self, inputs, out, recur):
        results = []

        labels = inputs['labels'][:, 2].detach().cpu()
        label_mask = inputs['label_mask'].detach().cpu()
        S2_mask = inputs["standalone_mask"].detach().cpu()
        gen_S2_joint = torch.zeros_like(out["gen_smpl_joint"].detach().cpu())
        gen_S2_joint[inputs["standalone_mask"].detach().cpu()>0] = out["gen_smpl_joint"][inputs["subaction_mask"]==2].detach().cpu()
        all_mask = inputs["valid_mask"].detach().cpu()
        gen_all_mesh = out["gen_smpl_mesh"].detach().cpu()
        gen_all_param = out["gen_smpl_param"].detach().cpu()
        gen_all_trans = out["gen_smpl_trans"].detach().cpu()
        gt_mesh = out["gt_smpl_mesh"].detach().cpu()

        frame_length = inputs["frame_length"].detach().cpu()

        S2_len = inputs['frame_length'][:, 2].detach().cpu()
        gen_S2_smpl_param = torch.zeros_like(out['gen_smpl_param'].detach().cpu())
        gen_S2_smpl_param[inputs["standalone_mask"].detach().cpu()>0] = out["gen_smpl_param"][inputs["subaction_mask"]==2].detach().cpu()
        gen_S2_smpl_trans = torch.zeros_like(out['gen_smpl_trans'].detach().cpu())
        gen_S2_smpl_trans[inputs["standalone_mask"].detach().cpu()>0] = out["gen_smpl_trans"][inputs["subaction_mask"]==2].detach().cpu()

        for i in range(len(labels)):
            label = int(labels[i])

            result = {
                'recur': recur,
                'label': label,
                'label_mask': label_mask[i],
                'S2_len': S2_len[i],
                'frame_length': frame_length[i].tolist(),
                'all_len': int(torch.sum(inputs["valid_mask"][i])),
                'all_mask': all_mask[i],
                'gt_mesh': gt_mesh[i],
                'gen_all_param': gen_all_param[i],
                'gen_all_trans': gen_all_trans[i],
                'gen_all_mesh': gen_all_mesh[i],
                'S2_mask': S2_mask[i],
                'gen_S2_smpl_param': gen_S2_smpl_param[i],
                'gen_S2_smpl_trans': gen_S2_smpl_trans[i],
            }
            results.append(result)

        return results



    def load_model(self, load_path):
        with open(load_path, "rb") as f:
            data = torch.load(f)

        self.model.load_state_dict(data)
