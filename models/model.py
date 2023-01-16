import torch
import torch.nn as nn
import copy
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d

from data.datautils.babel_label import label as BABEL_label
from data.datautils.babel_label import label_over_twenty
from data.datautils.transforms import *

from models.modules.transformer import CVAETransformerEncoder, CVAETransformerDecoder
from models.modules.postprocess import Postprocess
from models.modules.priornet import PriorNet

from utils.human_models import SMPLH
from utils.loss import ParamLoss, CoordLoss, KLLoss, AccelLoss

BABEL_label_rev = {v:k for k, v in BABEL_label.items()}
babel_over_twenty_numbers = sorted([BABEL_label[lab] for lab in label_over_twenty])
babel_from_zero = {numlab: i for i, numlab in enumerate(babel_over_twenty_numbers)}


class CVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.param_loss = ParamLoss(cfg.loss_type, cfg.loss_dim)
        self.coord_loss = CoordLoss(cfg.loss_type, cfg.loss_dim)
        self.kl_loss = KLLoss(cfg)
        self.accel_loss = AccelLoss()
        self.spec = cfg.Transformer_spec
        self.smpl_model = SMPLH(cfg)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_random_(p)

        batch_index = torch.tensor([[i]*cfg.max_input_len for i in range(60)]).reshape(-1)
        self.register_buffer("batch_index", batch_index, persistent=False)

        frame_index = torch.tensor([[j for j in range(cfg.max_input_len)] for i in range(60)]).reshape(-1)
        self.register_buffer("frame_index", frame_index, persistent=False)
        
        self.smplh_neutral_layer = copy.deepcopy(self.smpl_model.layer).cuda().requires_grad_(False)
        smplh_joint_regressor = self.smpl_model.joint_regressor.clone().detach().cuda()
        self.register_buffer("smplh_joint_regressor", smplh_joint_regressor, persistent=False)
        
        self.priornet = PriorNet(cfg, self.spec["embed_dim"])

        if cfg.encoder == "Transformer":
            self.encoder = CVAETransformerEncoder(cfg, self.spec["embed_dim"])
        else:
            raise NotImplementedError("unknown encoder")

        if cfg.decoder == "Transformer":
            self.decoder = CVAETransformerDecoder(cfg, self.spec["embed_dim"])
        else:
            raise NotImplementedError("unknown decoder")

        if cfg.postprocessor != "none":
            self.postprocessor = Postprocess(cfg)



    def forward(self, inputs, mode):
        # information from batch
        batch_size, input_duration = inputs["smpl_param"].shape[:2]

        smpl_trans = inputs["smpl_trans"]
        smpl_param = inputs["smpl_param"]
        label_mask = inputs["label_mask"]
        labels = inputs["labels"]
        frame_length = inputs["frame_length"]

        S1_end_mask = inputs["S1_end_mask"]
        valid_mask = inputs["valid_mask"]
        subaction_mask = inputs["subaction_mask"]
        output_mask = inputs["output_mask"]

        if mode != "gen":
            subaction_mean_mask = inputs["subaction_mean_mask"]

        if self.cfg.input_rotation_format == "6dim":
            input_param = matrix_to_rotation_6d(axis_angle_to_matrix(smpl_param.reshape(batch_size, input_duration, 52, 3))).reshape(batch_size, input_duration, 312)
        elif self.cfg.input_rotation_format == "axis":
            input_param = smpl_param
        
        input_trans = smpl_trans

        # pre-coding
        encoder_input = torch.cat((input_param, input_trans), dim=2).to(torch.float32)

        S1_end = encoder_input[S1_end_mask == 1].reshape(batch_size, self.cfg.S1_end_len, -1)

        # prior net: prior_mu, prior_logvar will be used in kl loss
        prior_mean, prior_logvar = self.priornet(S1_end, labels[:, 1:])

        # encoder
        if mode != "gen":
            posterior_mean, posterior_logvar = self.encoder(encoder_input, label_mask, valid_mask, subaction_mean_mask)

        if mode == "train":
            z = self.reparameterize(posterior_mean, posterior_logvar)
        elif mode == "gen":
            z = self.reparameterize(prior_mean, prior_logvar)

        if self.cfg.layered_pos_enc:
            transition_len = frame_length[:, 1]
        else:
            transition_len = None
        
        # decoder
        param_decoder_out, trans_decoder_out = self.decoder(z, (batch_size, input_duration, self.spec["embed_dim"]), output_mask, transition_len)
        
        if self.cfg.postprocessor != "none":
            # fill last frame pose into invalid timeframes
            if self.cfg.model_fill_last_frame:
                last_frame_index = torch.sum(frame_length, dim=1) - 1
                last_frame_param = param_decoder_out[torch.arange(0, batch_size), last_frame_index]
                last_frame_trans = trans_decoder_out[torch.arange(0, batch_size), last_frame_index]
                
                output_6d_param = last_frame_param.unsqueeze(1).expand((-1, input_duration, -1)).clone()
                output_trans = last_frame_trans.unsqueeze(1).expand((-1, input_duration, -1)).clone()

            else:
                output_6d_param = torch.zeros_like(smpl_param, dtype=torch.float32, device=smpl_param.device)
                output_trans = torch.zeros_like(smpl_trans, dtype=torch.float32, device=smpl_trans.device)
            # S1 mask 사용 대입
            s1_mask = subaction_mask == 0

            # 3_dim S1 -> 6_dim S1
            s1_smpl_6d = smpl_param[s1_mask].reshape(-1, 52, 3)
            s1_smpl_6d = axis_angle_to_matrix(s1_smpl_6d)
            s1_smpl_6d = matrix_to_rotation_6d(s1_smpl_6d).reshape(-1, 52*6)

            output_6d_param[s1_mask] = s1_smpl_6d
            output_trans[s1_mask] = smpl_trans[s1_mask]

            # decoder output에 output filter 씌워서 output 내용 추출
            # output param에 trs2 mask 씌워서 output 덮어씌울 위치 특정
            output_trs2_mask = output_mask > 0
            trs2_mask = (subaction_mask > 0)
            
            output_6d_param[trs2_mask] = param_decoder_out[output_trs2_mask]
            output_trans[trs2_mask] = trans_decoder_out[output_trs2_mask]

            # postprocessor
            output_6d_param, output_trans = self.postprocessor(output_6d_param, output_trans)
        
        else:
            # just append
            output_6d_param = torch.zeros((batch_size, input_duration, 312), dtype=torch.float32, device=smpl_param.device)
            output_trans = torch.zeros_like(smpl_trans, dtype=torch.float32, device=smpl_trans.device)
            output_trs2_mask = output_mask > 0
            trs2_mask = (subaction_mask > 0)
            
            output_6d_param[trs2_mask] = param_decoder_out[output_trs2_mask]
            output_trans[trs2_mask] = trans_decoder_out[output_trs2_mask]

        output_6d_param = output_6d_param.reshape(-1, 6)
        output_param = rot6d_to_axis_angle(output_6d_param)
        output_param = output_param.reshape(batch_size, self.cfg.max_input_len, -1)

        if self.cfg.postprocessor == "none":
            output_param[subaction_mask==0] = smpl_param[subaction_mask==0]
            output_trans[subaction_mask==0] = smpl_trans[subaction_mask==0]
        
        # gt body model parameters
        gt_body_params = {
            "global_orient": smpl_param[:, :, :3],
            "body_pose": smpl_param[:, :, 3:66],
            "hand_pose": smpl_param[:, :, 66:156],
            "transl": smpl_trans
        }

        # output body model parameters
        out_body_params = {
            "global_orient": output_param[:, :, :3],
            "body_pose": output_param[:, :, 3:66],
            "hand_pose": output_param[:, :, 66:156],
            "transl": output_trans
        }

        gt_smpl_mesh_cam = self.get_batch_smpl_mesh_cam(gt_body_params)
        gt_smpl_mesh_cam = gt_smpl_mesh_cam.reshape(batch_size, self.cfg.max_input_len, -1, 3)

        out_smpl_mesh_cam = self.get_batch_smpl_mesh_cam(out_body_params)
        out_smpl_mesh_cam = out_smpl_mesh_cam.reshape(batch_size, self.cfg.max_input_len, -1, 3)

        gt_smpl_joint_cam = torch.matmul(self.smplh_joint_regressor, gt_smpl_mesh_cam)  # batch * self.cfg.input_duration * self.smpl_model.orig_joint_num * 3, use joint regressor
        out_smpl_joint_cam = torch.matmul(self.smplh_joint_regressor, out_smpl_mesh_cam)  # batch * self.cfg.input_duration * self.smpl_model.orig_joint_num * 3, use joint regressor
    

        # generated transition outputs
        trans_out_smpl_joint_cam = torch.zeros_like(out_smpl_joint_cam)
        trans_out_smpl_joint_cam[output_mask==1] = out_smpl_joint_cam[subaction_mask==1]

        # gt transition outputs
        gt_trans_smpl_gt_cam = torch.zeros_like(gt_smpl_joint_cam)
        gt_trans_smpl_gt_cam[output_mask==1] = gt_smpl_joint_cam[subaction_mask==1]


        # wipe out invalid timeframe
        not_valid_mask = subaction_mask == -1
        gt_smpl_mesh_cam[not_valid_mask] = 0
        gt_smpl_joint_cam[not_valid_mask] = 0
        out_smpl_mesh_cam[not_valid_mask] = 0
        out_smpl_joint_cam[not_valid_mask] = 0

        accel_loss = self.accel_loss(out_smpl_joint_cam, valid_mask)

        loss = {}
        out = {}

        if mode == "train":
            loss['rec_pose'] = self.param_loss(
                torch.cat((output_param, output_trans), dim=2).to(torch.float32), 
                torch.cat((smpl_param, smpl_trans), dim=2).to(torch.float32),
                valid=valid_mask.unsqueeze(2).expand(-1, -1, 159)
            )
            loss['rec_mesh'] = self.coord_loss(out_smpl_mesh_cam, gt_smpl_mesh_cam) * self.cfg.mesh_weight
            loss['accel_loss'] = accel_loss * self.cfg.accel_weight
            loss['kl_loss'] = self.kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar) * self.cfg.kl_weight
        
        out['gen_smpl_mesh'] = out_smpl_mesh_cam
        out['gen_smpl_joint'] = out_smpl_joint_cam
        out['gt_smpl_mesh'] = gt_smpl_mesh_cam
        out['gt_smpl_joint'] = gt_smpl_joint_cam
        out['gen_smpl_param'] = output_param
        out['gen_smpl_trans'] = output_trans

        out['gt_transition_joint'] = gt_trans_smpl_gt_cam
        out['gen_transition_joint'] = trans_out_smpl_joint_cam

        return loss, out


    def generate_gaussian_prior(self, num_samples):
        mean = torch.zeros((num_samples, self.spec["embed_dim"]))
        logvar = torch.zeros((num_samples, self.spec["embed_dim"]))
        z = self.reparameterize(mean, logvar)
        return z

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    def get_batch_smpl_mesh_cam(self, body_params):
        smpl_out = self.smplh_neutral_layer(
            pose_body=body_params["body_pose"].reshape(-1, 63).to(torch.float32),
            pose_hand=body_params["hand_pose"].reshape(-1, 90).to(torch.float32),
            root_orient=body_params["global_orient"].reshape(-1, 3).to(torch.float32) if body_params["global_orient"] != None else None,
            trans=body_params["transl"].reshape(-1, 3).to(torch.float32)
        )
        return smpl_out.v



def get_model(cfg):
    if cfg.model == "CVAE":
        return CVAE(cfg)
    else:
        raise NotImplementedError("unknown model")
