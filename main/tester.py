import os
import torch
import numpy as np
import random

from torch.nn import DataParallel
from scipy import stats as st

from data.dataset import get_dataloader
from data.datautils.babel_label import label as BABEL_label
from data.datautils.babel_label import label_over_twenty, confusion
from data.datautils.transforms import *
from models.model import get_model

from utils.timer import Timer
from utils.vis import *
from utils.logger import colorLogger as Logger

from evaluate.load_classifier import load_classifier, load_classifier_for_fid
from evaluate.fid import calculate_frechet_distance

BABEL_label_rev = {v:k for k, v in BABEL_label.items()}

class Tester():
    def __init__(self, cfg, model=None, test_train_loader=None, test_logger=None):
        self.cfg = cfg
        # self.smplh = SMPLH(cfg)
        self.cfg.batch_size = self.cfg.test_batch_size

        if test_logger is not None:
            self.test_logger = test_logger
        else: 
            self.test_logger = Logger(cfg.log_dir, "test.log")

        if model is not None:
            self.model = model
        else:
            self.test_logger.debug("model loaded from " + cfg.test_model_path)
            self.model = DataParallel(get_model(cfg)).cuda()
            self.load_model(cfg.test_model_path)

        self.model.eval()
        
        self.test_timer = Timer()
        
        self.test_val_loader = get_dataloader(self.cfg, "val", drop_last=False, cap=cfg.test_per_label)

        if test_train_loader is not None:
            self.test_train_loader = test_train_loader
        else:
            if os.path.isfile(self.cfg.sampled_data_path):
                self.test_train_loader = get_dataloader(self.cfg, "train", sampled_file_path=self.cfg.sampled_data_path, drop_last=False, cap=cfg.test_per_label)
                self.test_logger.debug("loaded sampled training data from {}".format(self.cfg.sampled_data_path))

            else:
                self.test_logger.debug("failed to load sampled training data using randomly sampled training set.\nFID_train will be different from original result.")
                self.test_train_loader = get_dataloader(self.cfg, "train", drop_last=False, cap=cfg.test_per_label)
        
    def test(self):
        self.test_logger.debug("=================================================Evaluation result=================================================")
        test_num_rep = self.cfg.test_num_rep
        
        # generating samples
        gen_seen_label = []
        gen_seen_S2_joint = []
        gen_seen_S2_mask = []
        gen_unseen_label = []
        gen_unseen_S2_joint = []
        gen_unseen_S2_mask = []

        gt_seen_transition_joint = []
        gen_seen_transition_joint = []
        gt_unseen_transition_joint = []
        gen_unseen_transition_joint = []

        for rep in range(test_num_rep):
            torch.cuda.empty_cache()
            print("generating from previous motion in seen training set")
            gen_seen_label_, gen_seen_S2_joint_, gen_seen_S2_mask_, gt_seen_transition_joint_, gen_seen_transition_joint_ = self.generate_output("seen")

            print("generating from previous motion in unseen test set")
            gen_unseen_label_, gen_unseen_S2_joint_, gen_unseen_S2_mask_, gt_unseen_transition_joint_, gen_unseen_transition_joint_ = self.generate_output("unseen")

            gen_seen_label.append(gen_seen_label_)
            gen_seen_S2_joint.append(gen_seen_S2_joint_)
            gen_seen_S2_mask.append(gen_seen_S2_mask_)
            gen_unseen_label.append(gen_unseen_label_)
            gen_unseen_S2_joint.append(gen_unseen_S2_joint_)
            gen_unseen_S2_mask.append(gen_unseen_S2_mask_)

            gt_seen_transition_joint.append(gt_seen_transition_joint_)
            gen_seen_transition_joint.append(gen_seen_transition_joint_)
            gt_unseen_transition_joint.append(gt_unseen_transition_joint_)
            gen_unseen_transition_joint.append(gen_unseen_transition_joint_)

        seen_datas = (gt_seen_transition_joint, gen_seen_transition_joint, gen_seen_label)
        unseen_datas = (gt_unseen_transition_joint, gen_unseen_transition_joint, gen_unseen_label)

        self.run_evaluate_transition(None, None, seen_datas, unseen_datas)

        del self.test_train_loader
        print("sampling whole gt training set")
        train_gt_label, train_gt_S2_joint, train_gt_S2_mask = self.generate_all_gt("train")

        del self.test_val_loader
        print("sampling whole gt test set")
        val_gt_label, val_gt_S2_joint, val_gt_S2_mask = self.generate_all_gt("val")



        # Stats for real data
        print("evaluating gt statistics")

        train_fid_tr = []
        train_fid_test = []
        train_acc = []
        train_acc_5 = []
        train_div = []
        train_multimod = []

        test_fid_tr = []
        test_fid_test = []
        test_acc = []
        test_acc_5 = []
        test_div = []
        test_multimod = []
        
        # sample indices
        for rep in range(test_num_rep):
            print(rep, " / ", test_num_rep)
            cap = self.cfg.test_per_label
            train_gt_label_S2 = train_gt_label[:, 2]
            val_gt_label_S2 = val_gt_label[:, 2]

            train_indices = [i for i in range(len(train_gt_label_S2))]
            val_indices = [i for i in range(len(val_gt_label_S2))]

            train_sampled_indices = []
            val_sampled_indices = []
            random.shuffle(train_indices)
            random.shuffle(val_indices)

            label_list = [BABEL_label[l] for l in label_over_twenty]

            count = {l:0 for l in label_list}
            for idx in train_indices:
                cur_act_label = int(train_gt_label_S2[idx])
                if count[cur_act_label] < cap:
                    count[cur_act_label] += 1
                    train_sampled_indices.append(idx)
                
            count = {l:0 for l in label_list}
            for idx in val_indices:
                cur_act_label = int(val_gt_label_S2[idx])
                if count[cur_act_label] < cap:
                    count[cur_act_label] += 1
                    val_sampled_indices.append(idx)

            train_sampled_indices_for_gt = []
            val_sampled_indices_for_gt = []
            random.shuffle(train_indices)
            random.shuffle(val_indices)

            label_list = [BABEL_label[l] for l in label_over_twenty]

            count = {l:0 for l in label_list}
            for idx in train_indices:
                cur_act_label = int(train_gt_label_S2[idx])
                if count[cur_act_label] < cap:
                    count[cur_act_label] += 1
                    train_sampled_indices_for_gt.append(idx)
                
            count = {l:0 for l in label_list}
            for idx in val_indices:
                cur_act_label = int(val_gt_label_S2[idx])
                if count[cur_act_label] < cap:
                    count[cur_act_label] += 1
                    val_sampled_indices_for_gt.append(idx)

            # gen: tr, gt: tr
            tr_tr_stats = self.evaluate_S2_BABEL(train_gt_S2_joint[train_sampled_indices], train_gt_S2_mask[train_sampled_indices], train_gt_label[train_sampled_indices], 
                train_gt_S2_joint[train_sampled_indices_for_gt], train_gt_S2_mask[train_sampled_indices_for_gt], train_gt_label[train_sampled_indices_for_gt]
            )
            # gen: tr, gt: val
            val_tr_stats = self.evaluate_S2_BABEL(train_gt_S2_joint[train_sampled_indices], train_gt_S2_mask[train_sampled_indices], train_gt_label[train_sampled_indices],
                val_gt_S2_joint[val_sampled_indices_for_gt], val_gt_S2_mask[val_sampled_indices_for_gt], val_gt_label[val_sampled_indices_for_gt]
            )
            # gen: val, gt: tr
            tr_val_stats = self.evaluate_S2_BABEL(val_gt_S2_joint[val_sampled_indices], val_gt_S2_mask[val_sampled_indices], val_gt_label[val_sampled_indices], 
                train_gt_S2_joint[train_sampled_indices_for_gt], train_gt_S2_mask[train_sampled_indices_for_gt], train_gt_label[train_sampled_indices_for_gt]
            )
            # gen: val, gt: val
            val_val_stats = self.evaluate_S2_BABEL(val_gt_S2_joint[val_sampled_indices], val_gt_S2_mask[val_sampled_indices], val_gt_label[val_sampled_indices], 
                val_gt_S2_joint[val_sampled_indices_for_gt], val_gt_S2_mask[val_sampled_indices_for_gt], val_gt_label[val_sampled_indices_for_gt]
            )
            
            # (gt_acc, gen_acc, fid, float(div), float(multimod), gt_acc_5, gen_acc_5)

            train_fid_tr.append(tr_tr_stats[2])
            train_fid_test.append(val_tr_stats[2])
            train_acc.append(tr_tr_stats[1])
            train_acc_5.append(tr_tr_stats[5])
            train_div.append(tr_tr_stats[3])
            train_multimod.append(tr_tr_stats[4])

            test_fid_tr.append(tr_val_stats[2])
            test_fid_test.append(val_val_stats[2])
            test_acc.append(val_val_stats[1])
            test_acc_5.append(val_val_stats[5])
            test_div.append(val_val_stats[3])
            test_multimod.append(val_val_stats[4])

        train_fid_tr = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(train_fid_tr), scale=st.sem(train_fid_tr))
        train_fid_test = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(train_fid_test), scale=st.sem(train_fid_test))
        train_acc = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(train_acc), scale=st.sem(train_acc))
        train_acc_5 = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(train_acc_5), scale=st.sem(train_acc_5))
        train_div = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(train_div), scale=st.sem(train_div))
        train_multimod = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(train_multimod), scale=st.sem(train_multimod))

        test_fid_tr = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(test_fid_tr), scale=st.sem(test_fid_tr))
        test_fid_test = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(test_fid_test), scale=st.sem(test_fid_test))
        test_acc = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(test_acc), scale=st.sem(test_acc))
        test_acc_5 = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(test_acc_5), scale=st.sem(test_acc_5))
        test_div = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(test_div), scale=st.sem(test_div))
        test_multimod = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(test_multimod), scale=st.sem(test_multimod))


        real_train_sampled = {
            'fid_tr': train_fid_tr,
            'fid_test': train_fid_test,
            'acc': train_acc,
            'acc5': train_acc_5,
            'div': train_div,
            'multimod': train_multimod
        }

        real_test_sampled = {
            'fid_tr': test_fid_tr,
            'fid_test': test_fid_test,
            'acc': test_acc,
            'acc5': test_acc_5,
            'div': test_div,
            'multimod': test_multimod
        }

        # gen: tr, gt: tr
        tr_tr_stats = self.evaluate_S2_BABEL(train_gt_S2_joint, train_gt_S2_mask, train_gt_label, train_gt_S2_joint, train_gt_S2_mask, train_gt_label)
        # gen: tr, gt: val
        val_tr_stats = self.evaluate_S2_BABEL(train_gt_S2_joint, train_gt_S2_mask, train_gt_label, val_gt_S2_joint, val_gt_S2_mask, val_gt_label)
        # gen: val, gt: tr
        tr_val_stats = self.evaluate_S2_BABEL(val_gt_S2_joint, val_gt_S2_mask, val_gt_label, train_gt_S2_joint, train_gt_S2_mask, train_gt_label)
        # gen: val, gt: val
        val_val_stats = self.evaluate_S2_BABEL(val_gt_S2_joint, val_gt_S2_mask, val_gt_label, val_gt_S2_joint, val_gt_S2_mask, val_gt_label)

        real_train = {
            'fid_tr': tr_tr_stats[2],
            'fid_test': val_tr_stats[2],
            'acc': tr_tr_stats[1],
            'acc5': tr_tr_stats[5],
            'div': tr_tr_stats[3],
            'multimod': tr_tr_stats[4]
        }

        real_test = {
            'fid_tr': tr_val_stats[2],
            'fid_test': val_val_stats[2],
            'acc': val_val_stats[1],
            'acc5': val_val_stats[5],
            'div': val_val_stats[3],
            'multimod': val_val_stats[4]
        }

        # generated statistics
        print("evaluating generated statistics")
        seen_fid_tr = []
        seen_fid_test = []
        unseen_fid_tr = []
        unseen_fid_test = []
        seen_acc = []
        unseen_acc = []
        seen_div = []
        seen_multimod = []
        unseen_div = []
        unseen_multimod = []
        seen_acc_5 = []
        unseen_acc_5 = []

        for i in range(test_num_rep):
            print(i, " / ", test_num_rep)
            train_sampled_indices_for_gt = []
            val_sampled_indices_for_gt = []
            random.shuffle(train_indices)
            random.shuffle(val_indices)

            label_list = [BABEL_label[l] for l in label_over_twenty]

            count = {l:0 for l in label_list}
            for idx in train_indices:
                cur_act_label = int(train_gt_label_S2[idx])
                if count[cur_act_label] < cap:
                    count[cur_act_label] += 1
                    train_sampled_indices_for_gt.append(idx)
                
            count = {l:0 for l in label_list}
            for idx in val_indices:
                cur_act_label = int(val_gt_label_S2[idx])
                if count[cur_act_label] < cap:
                    count[cur_act_label] += 1
                    val_sampled_indices_for_gt.append(idx)

            seen_stats = self.evaluate_S2_BABEL(gen_seen_S2_joint[i], gen_seen_S2_mask[i], gen_seen_label[i], train_gt_S2_joint[train_sampled_indices_for_gt], train_gt_S2_mask[train_sampled_indices_for_gt], train_gt_label[train_sampled_indices_for_gt])
            seen_stats_test = self.evaluate_S2_BABEL(gen_seen_S2_joint[i], gen_seen_S2_mask[i], gen_seen_label[i], val_gt_S2_joint[val_sampled_indices_for_gt], val_gt_S2_mask[val_sampled_indices_for_gt], val_gt_label[val_sampled_indices_for_gt])
            unseen_stats = self.evaluate_S2_BABEL(gen_unseen_S2_joint[i], gen_unseen_S2_mask[i], gen_unseen_label[i], train_gt_S2_joint[train_sampled_indices_for_gt], train_gt_S2_mask[train_sampled_indices_for_gt], train_gt_label[train_sampled_indices_for_gt])
            unseen_stats_test = self.evaluate_S2_BABEL(gen_unseen_S2_joint[i], gen_unseen_S2_mask[i], gen_unseen_label[i], val_gt_S2_joint[val_sampled_indices_for_gt], val_gt_S2_mask[val_sampled_indices_for_gt], val_gt_label[val_sampled_indices_for_gt])

            seen_acc.append(seen_stats[1])
            seen_fid_tr.append(seen_stats[2])
            seen_div.append(seen_stats[3])
            seen_multimod.append(seen_stats[4])
            seen_fid_test.append(seen_stats_test[2])
            unseen_acc.append(unseen_stats[1])
            unseen_fid_tr.append(unseen_stats[2])
            unseen_div.append(unseen_stats[3])
            unseen_multimod.append(unseen_stats[4])
            unseen_fid_test.append(unseen_stats_test[2])
            seen_acc_5.append(seen_stats[6])
            unseen_acc_5.append(unseen_stats[6])
        
        seen_fid_tr = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(seen_fid_tr), scale=st.sem(seen_fid_tr))
        seen_fid_test = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(seen_fid_test), scale=st.sem(seen_fid_test))
        unseen_fid_tr = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(unseen_fid_tr), scale=st.sem(unseen_fid_tr))
        unseen_fid_test = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(unseen_fid_test), scale=st.sem(unseen_fid_test))
        seen_acc = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(seen_acc), scale=st.sem(seen_acc))
        unseen_acc = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(unseen_acc), scale=st.sem(unseen_acc))
        seen_acc_5 = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(seen_acc_5), scale=st.sem(seen_acc_5))
        unseen_acc_5 = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(unseen_acc_5), scale=st.sem(unseen_acc_5))
        seen_div = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(seen_div), scale=st.sem(seen_div))
        seen_multimod = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(seen_multimod), scale=st.sem(seen_multimod))
        unseen_div = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(unseen_div), scale=st.sem(unseen_div))
        unseen_multimod = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(unseen_multimod), scale=st.sem(unseen_multimod))

        seen_gen = {
            'fid_tr': seen_fid_tr,
            'fid_test': seen_fid_test,
            'acc': seen_acc,
            'acc5': seen_acc_5,
            'div': seen_div,
            'multimod': seen_multimod
        }

        unseen_gen = {
            'fid_tr': unseen_fid_tr,
            'fid_test': unseen_fid_test,
            'acc': unseen_acc,
            'acc5': unseen_acc_5,
            'div': unseen_div,
            'multimod': unseen_multimod
        }

        self.test_logger.debug("< Real_train_all >")
        self.test_logger.debug("fid_tr: {}, fid_test: {}, acc: {}, acc_5: {}, div: {}, multimod: {}".format(
            round(real_train['fid_tr'], 5),
            round(real_train['fid_test'], 5),
            round(real_train['acc'], 5),
            round(real_train['acc5'], 5),
            round(real_train['div'], 3),
            round(real_train['multimod'], 3),
        ))

        self.test_logger.debug("")

        self.test_logger.debug("< Real_test_all >")
        self.test_logger.debug("fid_tr: {}, fid_test: {}, acc: {}, acc_5: {}, div: {}, multimod: {}".format(
            round(real_test['fid_tr'], 5),
            round(real_test['fid_test'], 5),
            round(real_test['acc'], 5),
            round(real_test['acc5'], 5),
            round(real_test['div'], 3),
            round(real_test['multimod'], 3),
        ))
        self.test_logger.debug("")
        self.test_logger.debug("< Real_train_sampled >")
        self.test_logger.debug("fid_tr: {} +- {}, fid_test: {} +- {}, acc: {} +- {}, acc_5: {} +- {}, div: {} +- {}, multimod: {} +- {}".format(
            round((real_train_sampled['fid_tr'][0]+real_train_sampled['fid_tr'][1])/2, 5),
            round(-(real_train_sampled['fid_tr'][0]-real_train_sampled['fid_tr'][1])/2, 5),

            round((real_train_sampled['fid_test'][0]+real_train_sampled['fid_test'][1])/2, 5),
            round(-(real_train_sampled['fid_test'][0]-real_train_sampled['fid_test'][1])/2, 5),

            round((real_train_sampled['acc'][0]+real_train_sampled['acc'][1])/2, 5),
            round(-(real_train_sampled['acc'][0]-real_train_sampled['acc'][1])/2, 5),

            round((real_train_sampled['acc5'][0]+real_train_sampled['acc5'][1])/2, 5),
            round(-(real_train_sampled['acc5'][0]-real_train_sampled['acc5'][1])/2, 5),

            round((real_train_sampled['div'][0]+real_train_sampled['div'][1])/2, 3),
            round(-(real_train_sampled['div'][0]-real_train_sampled['div'][1])/2, 3),

            round((real_train_sampled['multimod'][0]+real_train_sampled['multimod'][1])/2, 3),
            round(-(real_train_sampled['multimod'][0]-real_train_sampled['multimod'][1])/2, 3),

        ))
        self.test_logger.debug("")

        self.test_logger.debug("< Real_test_sampled >")
        self.test_logger.debug("fid_tr: {} +- {}, fid_test: {} +- {}, acc: {} +- {}, acc_5: {} +- {}, div: {} +- {}, multimod: {} +- {}".format(
            round((real_test_sampled['fid_tr'][0]+real_test_sampled['fid_tr'][1])/2, 5),
            round(-(real_test_sampled['fid_tr'][0]-real_test_sampled['fid_tr'][1])/2, 5),

            round((real_test_sampled['fid_test'][0]+real_test_sampled['fid_test'][1])/2, 5),
            round(-(real_test_sampled['fid_test'][0]-real_test_sampled['fid_test'][1])/2, 5),

            round((real_test_sampled['acc'][0]+real_test_sampled['acc'][1])/2, 5),
            round(-(real_test_sampled['acc'][0]-real_test_sampled['acc'][1])/2, 5),

            round((real_test_sampled['acc5'][0]+real_test_sampled['acc5'][1])/2, 5),
            round(-(real_test_sampled['acc5'][0]-real_test_sampled['acc5'][1])/2, 5),

            round((real_test_sampled['div'][0]+real_test_sampled['div'][1])/2, 3),
            round(-(real_test_sampled['div'][0]-real_test_sampled['div'][1])/2, 3),

            round((real_test_sampled['multimod'][0]+real_test_sampled['multimod'][1])/2, 3),
            round(-(real_test_sampled['multimod'][0]-real_test_sampled['multimod'][1])/2, 3),
        ))
        self.test_logger.debug("")

        self.test_logger.debug("< Seen conditioned gen >")
        self.test_logger.debug("fid_tr: {} +- {}, fid_test: {} +- {}, acc: {} +- {}, acc_5: {} +- {}, div: {} +- {}, multimod: {} +- {}".format(
            round((seen_gen['fid_tr'][0]+seen_gen['fid_tr'][1])/2, 5),
            round(-(seen_gen['fid_tr'][0]-seen_gen['fid_tr'][1])/2, 5),

            round((seen_gen['fid_test'][0]+seen_gen['fid_test'][1])/2, 5),
            round(-(seen_gen['fid_test'][0]-seen_gen['fid_test'][1])/2, 5),

            round((seen_gen['acc'][0]+seen_gen['acc'][1])/2, 5),
            round(-(seen_gen['acc'][0]-seen_gen['acc'][1])/2, 5),

            round((seen_gen['acc5'][0]+seen_gen['acc5'][1])/2, 5),
            round(-(seen_gen['acc5'][0]-seen_gen['acc5'][1])/2, 5),

            round((seen_gen['div'][0]+seen_gen['div'][1])/2, 3),
            round(-(seen_gen['div'][0]-seen_gen['div'][1])/2, 3),

            round((seen_gen['multimod'][0]+seen_gen['multimod'][1])/2, 3),
            round(-(seen_gen['multimod'][0]-seen_gen['multimod'][1])/2, 3),

        ))
        self.test_logger.debug("")
        self.test_logger.debug("< Unseen conditioned gen >")
        self.test_logger.debug("fid_tr: {} +- {}, fid_test: {} +- {}, acc: {} +- {}, acc_5: {} +- {}, div: {} +- {}, multimod: {} +- {}".format(
            round((unseen_gen['fid_tr'][0]+unseen_gen['fid_tr'][1])/2, 5),
            round(-(unseen_gen['fid_tr'][0]-unseen_gen['fid_tr'][1])/2, 5),

            round((unseen_gen['fid_test'][0]+unseen_gen['fid_test'][1])/2, 5),
            round(-(unseen_gen['fid_test'][0]-unseen_gen['fid_test'][1])/2, 5),

            round((unseen_gen['acc'][0]+unseen_gen['acc'][1])/2, 5),
            round(-(unseen_gen['acc'][0]-unseen_gen['acc'][1])/2, 5),

            round((unseen_gen['acc5'][0]+unseen_gen['acc5'][1])/2, 5),
            round(-(unseen_gen['acc5'][0]-unseen_gen['acc5'][1])/2, 5),

            round((unseen_gen['div'][0]+unseen_gen['div'][1])/2, 3),
            round(-(unseen_gen['div'][0]-unseen_gen['div'][1])/2, 3),

            round((unseen_gen['multimod'][0]+unseen_gen['multimod'][1])/2, 3),
            round(-(unseen_gen['multimod'][0]-unseen_gen['multimod'][1])/2, 3),

        ))
        self.test_logger.debug("===================================================================================================================")
        return



    def run_evaluate_transition(self, train_data=None, val_data=None, seen_datas=None, unseen_datas=None):
        self.test_logger.info("==================================================Transition Test Start=================================================")
        torch.cuda.empty_cache()
        test_num_rep = self.cfg.test_num_rep
        
        if seen_datas is None:
            # generate data like test_BABEL_cond
            exit(0)

        # (train_gt_all_joint, train_gt_subaction_mask, train_gt_output_mask, train_gt_label) = train_data
        # (val_gt_all_joint, val_gt_subaction_mask, val_gt_output_mask, val_gt_label) = val_data
        (gt_seen_transition_joints, gen_seen_transition_joints, _) = seen_datas
        (gt_unseen_transition_joints, gen_unseen_transition_joints, _) = unseen_datas

        fid_gt_gen_seen = []
        fid_gt_gen_unseen = []
        fid_test_gt_gen_seen = []
        fid_test_gt_gen_unseen = []


        # for ssmct
        # with open(os.path.join(self.cfg.model_out_dir, "ssmct_gen_seen_trans_joint"), "rb") as f:
        #     gen_seen_transition_joints = pickle.load(f)
        # with open(os.path.join(self.cfg.model_out_dir, "ssmct_gen_unseen_trans_joint"), "rb") as f:
        #     gen_unseen_transition_joints = pickle.load(f)



        for i in range(len(gt_seen_transition_joints)):
            gt_seen_transition_joint = gt_seen_transition_joints[i]
            gen_seen_transition_joint = gen_seen_transition_joints[i]
            gt_unseen_transition_joint = gt_unseen_transition_joints[i]
            gen_unseen_transition_joint = gen_unseen_transition_joints[i]

            fid_gt_gen_seen_ = self.evaluate_transition_BABEL(gen_seen_transition_joint, gt_seen_transition_joint)
            fid_test_gt_gen_seen_ = self.evaluate_transition_BABEL(gen_seen_transition_joint, gt_unseen_transition_joint)

            fid_gt_gen_unseen_ = self.evaluate_transition_BABEL(gen_unseen_transition_joint, gt_seen_transition_joint)
            fid_test_gt_gen_unseen_ = self.evaluate_transition_BABEL(gen_unseen_transition_joint, gt_unseen_transition_joint)

            fid_gt_gen_seen.append(fid_gt_gen_seen_)
            fid_gt_gen_unseen.append(fid_gt_gen_unseen_)

            fid_test_gt_gen_seen.append(fid_test_gt_gen_seen_)
            fid_test_gt_gen_unseen.append(fid_test_gt_gen_unseen_)


        fid_gt_gen_seen = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(fid_gt_gen_seen), scale=st.sem(fid_gt_gen_seen))
        fid_gt_gen_unseen = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(fid_gt_gen_unseen), scale=st.sem(fid_gt_gen_unseen))

        fid_test_gt_gen_seen = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(fid_test_gt_gen_seen), scale=st.sem(fid_test_gt_gen_seen))
        fid_test_gt_gen_unseen = st.t.interval(alpha=0.95, df=test_num_rep-1, loc=np.mean(fid_test_gt_gen_unseen), scale=st.sem(fid_test_gt_gen_unseen))

        self.test_logger.debug("< gt ~ seen_ours >")
        self.test_logger.debug("FID_tr: {} +- {}, FID_test: {} +- {}".format(
            round((fid_gt_gen_seen[1] + fid_gt_gen_seen[0])/2, 4), 
            round((fid_gt_gen_seen[1] - fid_gt_gen_seen[0])/2, 4), 
            round((fid_test_gt_gen_seen[1] + fid_test_gt_gen_seen[0])/2, 4), 
            round((fid_test_gt_gen_seen[1] - fid_test_gt_gen_seen[0])/2, 4), 
        ))
        self.test_logger.debug("")

        self.test_logger.debug("< gt ~ unseen_ours >")
        self.test_logger.debug("FID_tr: {} +- {}, FID_test: {} +- {}".format(
            round((fid_gt_gen_unseen[1] + fid_gt_gen_unseen[0])/2, 4), 
            round((fid_gt_gen_unseen[1] - fid_gt_gen_unseen[0])/2, 4), 
            round((fid_test_gt_gen_unseen[1] + fid_test_gt_gen_unseen[0])/2, 4), 
            round((fid_test_gt_gen_unseen[1] - fid_test_gt_gen_unseen[0])/2, 4), 
        ))
        self.test_logger.debug("")
        return


    def generate_output(self, split="unseen"):
        if split == "seen":
            loader = self.test_train_loader
        elif split == "unseen":
            loader = self.test_val_loader
        
        gen_label = None
        gen_S2_joint = None
        gen_S2_mask = None
        gt_transition_joint = None
        gen_transition_joint = None

        for itr, inputs in enumerate(loader):
            with torch.no_grad():
                _, out = self.model(inputs, "gen")
            if gen_S2_joint == None:
                gen_S2_joint = torch.zeros_like(out["gen_smpl_joint"].detach().cpu())
                gen_S2_joint[inputs["standalone_mask"].detach().cpu()>0] = out["gen_smpl_joint"][inputs["subaction_mask"]==2].detach().cpu()
                gen_S2_mask = inputs["standalone_mask"].detach().cpu()
                gen_label = inputs["labels"].detach().cpu()

                gt_transition_joint = out["gt_transition_joint"].detach().cpu()
                gen_transition_joint = out["gen_transition_joint"].detach().cpu()
            
            else:
                gen_S2_joint_ = torch.zeros_like(out["gen_smpl_joint"].detach().cpu())
                gen_S2_joint_[inputs["standalone_mask"].detach().cpu()>0] = out["gen_smpl_joint"][inputs["subaction_mask"]==2].detach().cpu()
                gen_S2_mask_ = inputs["standalone_mask"].detach().cpu()
                gen_label_ = inputs["labels"].detach().cpu()

                gen_S2_joint = torch.cat((gen_S2_joint, gen_S2_joint_), dim=0)
                gen_S2_mask = torch.cat((gen_S2_mask, gen_S2_mask_), dim=0)
                gen_label = torch.cat((gen_label, gen_label_), dim=0)

                gt_transition_joint = torch.cat((gt_transition_joint, out["gt_transition_joint"].detach().cpu()), dim=0)
                gen_transition_joint = torch.cat((gen_transition_joint, out["gen_transition_joint"].detach().cpu()), dim=0)

        return gen_label, gen_S2_joint, gen_S2_mask, gt_transition_joint, gen_transition_joint


    def generate_all_gt(self, split="train"):
        assert split in ["train", "val"]

        # not only filtered, but gt including all sampled pairs
        if split == "train":
            loader = get_dataloader(self.cfg, "train", filter_label="twenty", drop_last=False, sampled_file_path=self.cfg.sampled_data_path, cap=self.cfg.test_per_label)
        elif split == "val":
            loader = get_dataloader(self.cfg, "val", filter_label="twenty", drop_last=False)

        gt_label = None
        gt_S2_joint = None
        gt_S2_mask = None


        for itr, inputs in enumerate(loader):
            with torch.no_grad():
                _, out = self.model(inputs, "gen")
            gt_label_ = inputs["labels"].cpu()
            gt_S2_joint_ = torch.zeros_like(out["gt_smpl_joint"])
            gt_S2_joint_[inputs["standalone_mask"]>0] = out["gt_smpl_joint"][inputs["subaction_mask"]==2]
            gt_S2_joint_ = gt_S2_joint_.cpu()
            gt_S2_mask_ = inputs["standalone_mask"].cpu()

            if gt_label == None:
                gt_label = gt_label_
                gt_S2_joint = gt_S2_joint_
                gt_S2_mask = gt_S2_mask_
            else:
                gt_label = torch.cat((gt_label, gt_label_), dim=0)
                gt_S2_joint = torch.cat((gt_S2_joint, gt_S2_joint_), dim=0)
                gt_S2_mask = torch.cat((gt_S2_mask, gt_S2_mask_), dim=0)
        del loader

        return gt_label, gt_S2_joint, gt_S2_mask


    def evaluate_S2_BABEL(self, gen_joint, gen_mask, gen_label, gt_joint, gt_mask, gt_label):
        """
            Inputs
            gen_joint (num_gen * maxlen * 52 * 3)
            gen_mask (num_gen * maxlen)
            gen_label (num_gen * 3)
            gt_joint (num_gt * maxlen * 52 * 3)
            gt_mask (num_gen * maxlen)
            gt_label (num_gt * 3)

            Outputs
            stat (dict) -> metrics
        """
        torch.cuda.empty_cache()

        if self.cfg.action_dataset == "BABEL":
            gt_label = gt_label[:, 2]
            gen_label = gen_label[:, 2]
            gt_label = [BABEL_label_rev[int(lab)] for lab in gt_label]
            gen_label = [BABEL_label_rev[int(lab)] for lab in gen_label]
            gt_label = [label_over_twenty.index(lab) for lab in gt_label]
            gen_label = [label_over_twenty.index(lab) for lab in gen_label]
            gt_label = torch.tensor(gt_label, device=torch.device("cuda:0"))
            gen_label = torch.tensor(gen_label, device=torch.device("cuda:0"))

        # load classifier
        classifier = load_classifier(self.cfg, torch.device("cuda:0"))
        feature_extractor = load_classifier_for_fid(self.cfg, torch.device("cuda:0"))

        gt_joint = gt_joint.reshape((gt_joint.shape[0], gt_joint.shape[1], -1)).cuda()
        gen_joint = gen_joint.reshape((gen_joint.shape[0], gen_joint.shape[1], -1)).cuda()

        # extract representation and prediction
        
        with torch.no_grad():
            gt_prob = classifier(gt_joint)
            gt_pred = gt_prob.max(dim=1).indices
            # gt_correct = gt_pred == gt_label
            gt_correct = self.get_correct_pred_conf(gt_pred, gt_label)
            gt_acc = float(gt_correct / gt_label.shape[0])

            gt_pred_5 = torch.topk(gt_prob, k=5, dim=1).indices
            gt_label_5 = gt_label.unsqueeze(1).expand((gt_label.shape[0], 5))
            # gt_correct_5 = gt_pred_5 == gt_label_5
            gt_correct_5 = self.get_correct_top5_pred_conf(gt_pred_5, gt_label_5)
            gt_acc_5 = float(gt_correct_5 / gt_label.shape[0])

            gen_prob = classifier(gen_joint)
            gen_pred = gen_prob.max(dim=1).indices
            # gen_correct = gen_pred == gen_label
            gen_correct = self.get_correct_pred_conf(gen_pred, gen_label)
            gen_acc = float(gen_correct / gen_label.shape[0])

            gen_pred_5 = torch.topk(gen_prob, k=5, dim=1).indices
            gen_label_5 = gen_label.unsqueeze(1).expand((gen_label.shape[0], 5))
            # gen_correct_5 = gen_pred_5 == gen_label_5
            gen_correct_5 = self.get_correct_top5_pred_conf(gen_pred_5, gen_label_5)
            gen_acc_5 = float(gen_correct_5 / gen_label.shape[0])

            gt_rep = feature_extractor(gt_joint)
            gen_rep = feature_extractor(gen_joint)
            
            gt_dist = self.estimate_distribution(gt_rep)
            gen_dist = self.estimate_distribution(gen_rep)

            fid = self.calculate_fid(gt_dist, gen_dist)
            div, multimod = self.calculate_diversity_multimodality(gen_rep, gen_label, len(label_over_twenty) if self.cfg.action_dataset == "BABEL" else 12)

        return (gt_acc, gen_acc, fid, float(div), float(multimod), gt_acc_5, gen_acc_5)


    def evaluate_transition_BABEL(self, gen_joint, gt_joint):
        torch.cuda.empty_cache()
        feature_extractor = load_classifier_for_fid(self.cfg, torch.device("cuda:0"))

        gt_joint = gt_joint.reshape((gt_joint.shape[0], gt_joint.shape[1], -1)).cuda()
        gen_joint = gen_joint.reshape((gen_joint.shape[0], gen_joint.shape[1], -1)).cuda()
        gt_rep = feature_extractor(gt_joint)
        gen_rep = feature_extractor(gen_joint)
        
        gt_dist = self.estimate_distribution(gt_rep)
        gen_dist = self.estimate_distribution(gen_rep)
        fid = self.calculate_fid(gt_dist, gen_dist)

        return fid


    def get_correct_pred_conf(self, pred, label):
        correct = 0
        for i in range(len(label)):
            gt_label_idx = int(label[i])
            predicted_idx = int(pred[i])

            gt_label = label_over_twenty[gt_label_idx]
            predicted = label_over_twenty[predicted_idx]

            if gt_label in confusion.keys():
                confusion_list = confusion[gt_label]
                if predicted in confusion_list:
                    correct += 1

            else:
                if gt_label_idx == predicted_idx:
                    correct += 1

        return correct


    def get_correct_top5_pred_conf(self, pred, label):
        correct = 0
        for i in range(len(label)):
            gt_label_idx = int(label[i][0])
            predicted_top5_idx = [int(idx) for idx in pred[i]]

            gt_label = label_over_twenty[gt_label_idx]
            predicted_top5 = [label_over_twenty[idx] for idx in predicted_top5_idx]

            if gt_label in confusion.keys():
                confusion_list = confusion[gt_label]
                in_confusion = False

                for predicted in predicted_top5:
                    if predicted in confusion_list:
                        in_confusion = True

                if in_confusion:
                    correct += 1

            else:
                if gt_label_idx in predicted_top5_idx:
                    correct += 1

        return correct


    def estimate_distribution(self, inputs):
        inputs = inputs.cpu().detach().numpy()
        mu = np.mean(inputs, axis=0)
        sigma = np.cov(inputs, rowvar=False)
        return mu, sigma
    

    def calculate_fid(self, statistics_1, statistics_2):
        return calculate_frechet_distance(statistics_1[0], statistics_1[1], statistics_2[0], statistics_2[1])


    def calculate_diversity_multimodality(self, activations, labels, num_labels):
        diversity_times = 200
        multimodality_times = 20
        labels = labels.long()
        num_motions = len(labels)

        num_real_labels = len(labels.unique())

        diversity = 0
        first_indices = np.random.randint(0, num_motions, diversity_times)
        second_indices = np.random.randint(0, num_motions, diversity_times)
        for first_idx, second_idx in zip(first_indices, second_indices):
            diversity += torch.dist(activations[first_idx, :],
                                    activations[second_idx, :])
        diversity /= diversity_times

        multimodality = 0
        labal_quotas = np.repeat(multimodality_times, num_labels)
        while np.any(labal_quotas > 0):
            first_idx = np.random.randint(0, num_motions)
            first_label = labels[first_idx]
            if not labal_quotas[first_label]:
                continue

            second_idx = np.random.randint(0, num_motions)
            second_label = labels[second_idx]
            while first_label != second_label:
                second_idx = np.random.randint(0, num_motions)
                second_label = labels[second_idx]

            labal_quotas[first_label] -= 1

            first_activation = activations[first_idx, :]
            second_activation = activations[second_idx, :]
            multimodality += torch.dist(first_activation,
                                        second_activation)
            
            if np.sum(labal_quotas) == (num_labels - num_real_labels) * multimodality_times:
                break

        multimodality /= (multimodality_times * num_labels)

        return diversity, multimodality


    def load_model(self, load_path):
        with open(load_path, "rb") as f:
            data = torch.load(f)

        self.model.load_state_dict(data)
