import os
import torch
import imageio
import numpy as np
import pickle
# import time

from torch.optim import AdamW
from torch.nn import DataParallel

# from main.tester import Tester
from data.dataset import get_dataloader
from data.datautils.babel_label import label as BABEL_label

from models.model import get_model

from utils.timer import Timer, sec2minhrs
from utils.logger import colorLogger as Logger
from utils.vis import *
from utils.human_models import SMPLH

BABEL_label_rev = {v:k for k, v in BABEL_label.items()}

hyperparams = [
    # overall hyperparams
    "model",

    # training hyperparams
    "batch_size",
    "kl_weight", 
    "train_lr",

    # model hyperparams
    "msg",
]

class Trainer():
    def __init__(self, cfg):

        self.cfg = cfg
        self.action_dataloader = get_dataloader(cfg, "train", cap=cfg.train_per_label)

        self.model = DataParallel(get_model(cfg)).cuda()

        if cfg.continue_train:
            self.load_model(cfg.load_model_path)
        self.optimizer = AdamW(self.model.parameters(), cfg.train_lr)

        self.train_timer = Timer()
        self.logger = Logger(cfg.log_dir)
        # self.test_logger = Logger(cfg.log_dir, "test.log")

    def train(self):

        self.logger.info("===================================================Training start===================================================")
        # args printing here
        self.logger.info("Using {} action training data".format(len(self.action_dataloader.dataset)))
        self.logger.debug("All settings used:")

        for k in hyperparams:
            self.logger.debug("{}: {}".format(k, vars(self.cfg)[k]))

        self.logger.info("====================================================================================================================")
        self.logger.info("")
        
        self.save_model("init.pkl")
        
        with open(os.path.join(self.cfg.model_out_dir, "sampled_training_data.pkl"), "wb") as f:
            pickle.dump(self.action_dataloader.dataset.dbs[0].sampled, f)
            print("Sampled dataset saved as ", os.path.join(self.cfg.model_out_dir, "sampled_training_data.pkl"))
            setattr(self.cfg, "sampled_data_path", os.path.join(self.cfg.model_out_dir, "sampled_training_data.pkl"))

        self.train_timer.tic()
        for epoch in range(self.cfg.start_epoch, self.cfg.end_epoch):
            self.model.train()
            is_last = (epoch == self.cfg.end_epoch - 1)
            epoch_loss = {
                "rec_pose": torch.tensor(0), 
                "rec_mesh": torch.tensor(0), 
                "kl_loss": torch.tensor(0),
                "accel_loss": torch.tensor(0)}

            for itr, inputs in enumerate(self.action_dataloader):
                # forward
                self.optimizer.zero_grad()
                self.model.module.zero_grad()
                loss, out = self.model(inputs, 'train')

                loss = {k: loss[k].mean() for k in loss}
                epoch_loss = {k: loss[k] + epoch_loss[k] for k in loss}

                loss = sum(loss[k] for k in loss)
                loss.backward()

                self.optimizer.step()
            
            if (epoch % self.cfg.print_epoch == 0 and epoch != 0) or is_last:
                epoch_loss = {k: epoch_loss[k]/(itr+1) for k in epoch_loss}
                total_time, avg_time = self.train_timer.toc()
                ETA = (avg_time * (self.cfg.end_epoch - self.cfg.start_epoch)) / self.cfg.print_epoch
                ETA = ETA - total_time

                h, m, s = sec2minhrs(ETA)
                h2, m2, s2 = sec2minhrs(total_time)
                print("epoch: {}, avg_time: {} s/epoch, elapsed_time: {} h {} m {} s, ETA: {} h {} m {} s"
                      .format(epoch, round(avg_time / self.cfg.print_epoch, 4), h2, m2, s2, h, m, s))
                self.logger.debug("")
                self.logger.debug("epoch: {}, avg_time: {} s/epoch, elapsed_time: {} h {} m {} s, ETA: {} h {} m {} s"
                    .format(epoch, round(avg_time / self.cfg.print_epoch, 4), h2, m2, s2, h, m, s))
                self.logger.debug(
                    "rec_pose: {}, rec_mesh: {}, kl_loss: {}, accel_loss: {}".format(
                        # round(float(epoch_loss["rec_rot"]), 5),
                        round(float(epoch_loss["rec_pose"]), 5),
                        round(float(epoch_loss["rec_mesh"] / self.cfg.mesh_weight), 5),
                        # round(float(epoch_loss["rec_disp"]), 5),
                        # round(float(epoch_loss["motion"]), 4),
                        round(float(epoch_loss["kl_loss"] / self.cfg.kl_weight), 4),
                        round(float(epoch_loss["accel_loss"] / self.cfg.accel_weight), 6),
                    ))
            
            if (epoch % self.cfg.vis_epoch == 0 and epoch != 0) or is_last:
                if self.cfg.model == "CVAE":
                    action = inputs["label_mask"][:self.cfg.vis_num]
                    valid_mask = inputs["valid_mask"][:self.cfg.vis_num]
                    label_texts = [[BABEL_label_rev[int(label_idx)] for label_idx in seq] for seq in action]

                    gt_mesh = out["gt_smpl_mesh"][:self.cfg.vis_num].detach().cpu()
                    gen_mesh = out["gen_smpl_mesh"][:self.cfg.vis_num].detach().cpu()

                    names = range(1, self.cfg.vis_num+1)
                    names = [str(name) for name in names]
                    gt_paths = [os.path.join(self.cfg.vis_dir, "train_vis", str(epoch), "gt") for pth in names]
                    gen_paths = [os.path.join(self.cfg.vis_dir, "train_vis", str(epoch), "gen") for pth in names]

                    visualize_batch(self.cfg, gen_mesh, valid_mask, gen_paths, names, label_texts, method="render")
                    visualize_batch(self.cfg, gt_mesh, valid_mask, gt_paths, names, label_texts, method="render")

                self.save_model("last.pkl")

        return


    def get_state_dict(self, model):
        dump_key = []
        for k in model.state_dict():
            if 'smpl_layer' in k:
                dump_key.append(k)

        update_key = [k for k in model.state_dict().keys() if k not in dump_key]
        return {k: model.state_dict()[k] for k in update_key}

    def save_model(self, name, dump_smpl=True):
        save_path = os.path.join(self.cfg.model_out_dir, name)
        dump_key = []
        if dump_smpl:
            for k in self.model.state_dict():
                if 'smpl_layer' in k:
                    dump_key.append(k)

        update_key = [k for k in self.model.state_dict().keys() if k not in dump_key]
        update_dict = {k: self.model.state_dict()[k] for k in update_key}

        torch.save(update_dict, save_path)
        return

    def visualize_batch(self, meshes, masks, paths, names, labels):
        faces = SMPLH(self.cfg).face
        speed_sec = {'duration': 0.05}

        batch_size, duration = meshes.shape[:2]

        for mesh, mask, path, name, label in zip(meshes, masks, paths, names, labels):

            # vis_img = vis_motion(mesh, faces)
            vis_img = vis_motion_vertices(mesh, faces)

            img_lst = []
            for j in range(duration):
                if int(mask[j]) == 0:
                    continue
                l = label[j]
                i = vis_img[j]
                cv2.putText(i, l, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                img_lst.append(i.astype(np.uint8))

            os.makedirs(path, exist_ok=True)
            imageio.mimsave(os.path.join(path, name+".gif"), img_lst, **speed_sec)
                

    def load_model(self, load_path):
        with open(load_path, "rb") as f:
            data = torch.load(f)

        self.model.load_state_dict(data)