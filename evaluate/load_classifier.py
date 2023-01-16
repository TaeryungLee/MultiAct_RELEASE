import torch
import math
from evaluate.motion_gan import MotionDiscriminator
from torch import nn
from data.datautils.babel_label import label_over_twenty
from data.datautils.babel_label import label as BABEL_label

BABEL_label_inv = {v:k for k, v in BABEL_label.items()}

from torch.optim import AdamW


classifier_model_files = {
    'BABEL': './evaluate/model_file/action_recognition_model_babel.tar'
}


def train_classifier(cfg, device, gt_joint, gt_mask, gt_label, val_joint, val_mask, val_label):
    gt_label = gt_label[:, 2]
    val_label = val_label[:, 2]
    gt_label = [BABEL_label_inv[int(lab)] for lab in gt_label]
    val_label = [BABEL_label_inv[int(lab)] for lab in val_label]
    gt_label = [label_over_twenty.index(lab) for lab in gt_label]
    val_label = [label_over_twenty.index(lab) for lab in val_label]

    gt_label = torch.tensor(gt_label, device=device)
    val_label = torch.tensor(val_label, device=device)

    batch_size = cfg.batch_size
    num_train_data = gt_joint.shape[0]
    num_train_iter = math.ceil(num_train_data/batch_size)
    num_val_data = val_joint.shape[0]
    num_val_iter = math.ceil(num_val_data/batch_size)

    num_epoch = 301
    dataset_name = cfg.action_dataset
    body_model = cfg.body_model
    input_size = 52 * 3 if body_model == "SMPLH" else 24 * 3

    # print(unique_labels_dict)
    # print(dataset_name, body_model, input_size)
    classifier = MotionDiscriminator(input_size=input_size, hidden_size=256, hidden_layer=4, output_size=len(label_over_twenty)).to(gt_joint.device)
    classifier.train()
    celoss = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=0.0005)

    best_val = -1
    for epoch in range(num_epoch):
        right = 0
        wrong = 0
        for itr in range(num_train_iter):
            classifier.zero_grad()
            itr_start = itr * batch_size
            itr_end = min(num_train_data, (itr+1)*batch_size)
            batch_joint = gt_joint[itr_start:itr_end].reshape((itr_end-itr_start, -1, input_size))
            batch_mask = gt_mask[itr_start:itr_end]
            batch_label = gt_label[itr_start:itr_end]
            
            out = classifier(batch_joint)
            pred = out.max(dim=1).indices
            corr = pred == batch_label
            wr = pred != batch_label

            right += int(sum(corr))
            wrong += int(sum(wr))

            loss = celoss(out, batch_label)
            loss.backward()
            optimizer.step()
        
        print("train", epoch, round(right / (right+wrong), 2))
        if epoch % 5 != 0 or epoch < 100:
            continue

        val_right = 0
        val_wrong = 0
        with torch.no_grad():
            for itr in range(num_val_iter):
                itr_start = itr * batch_size
                itr_end = min(num_val_data, (itr+1)*batch_size)
                batch_joint = val_joint[itr_start:itr_end].reshape((itr_end-itr_start, -1, input_size))
                batch_mask = val_mask[itr_start:itr_end]
                batch_label = val_label[itr_start:itr_end]
                
                out = classifier(batch_joint)
                pred = out.max(dim=1).indices
                corr = pred == batch_label
                wr = pred != batch_label

                val_right += int(sum(corr))
                val_wrong += int(sum(wr))

            print("===============")
            print("val", round(val_right / (val_right+val_wrong), 2))
            print("===============")

        if best_val <= (0.01*right/(right+wrong) + val_right/(val_right+val_wrong)):
            print(">>> current best <<<")
            print(round(right/(right+wrong), 2))
            print(round(val_right/(val_right+val_wrong), 2))
            best_val = (0.01*right/(right+wrong) + val_right/(val_right+val_wrong))
            with open(classifier_model_files[dataset_name], "wb") as f:
                torch.save(classifier.state_dict(), f)

    return



def load_classifier(cfg, device):
    dataset_name = cfg.action_dataset
    body_model = cfg.body_model
    num_joints = 52 if body_model == "SMPLH" else 24

    model = torch.load(classifier_model_files[dataset_name])
    classifier = MotionDiscriminator(num_joints * 3, 256 if body_model=="SMPLH" else 128, 4 if body_model=="SMPLH" else 2, len(label_over_twenty) if body_model=="SMPLH" else len(cfg.action_profiles)).to(device)
    if 'model' in list(model.keys()):
        classifier.load_state_dict(model['model'])
    else:
        classifier.load_state_dict(model)
    classifier.eval()

    return classifier


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            # motion sequence: batch * length * (joint*3)
            init_root = motion_sequence[:, :1, :3]  # batch * 1 * 3
            init_root = init_root.expand(motion_sequence.shape[0], motion_sequence.shape[1], -1)
            init_root = init_root.repeat(1, 1, int(motion_sequence.shape[2]/3))
            motion_sequence = motion_sequence - init_root
            motion_sequence = motion_sequence.permute(1, 0, 2)
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        return lin1


def load_classifier_for_fid(cfg, device):
    dataset_name = cfg.action_dataset
    body_model = cfg.body_model
    num_joints = 52 if body_model == "SMPLH" else 24

    model = torch.load(classifier_model_files[cfg.action_dataset])
    classifier = MotionDiscriminatorForFID(num_joints * 3, 256 if body_model=="SMPLH" else 128, 4 if body_model=="SMPLH" else 2, len(label_over_twenty) if body_model=="SMPLH" else len(cfg.action_profiles)).to(device)
    if 'model' in list(model.keys()):
        classifier.load_state_dict(model['model'])
    else:
        classifier.load_state_dict(model)
    classifier.eval()

    return classifier
