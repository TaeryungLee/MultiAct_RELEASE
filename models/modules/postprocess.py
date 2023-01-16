from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle, matrix_to_euler_angles, matrix_to_rotation_6d, rotation_6d_to_matrix
import torch.nn as nn
import torch


class Postprocess(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.output_rotation_format == "6dim":
            input_dim = 6
        else:
            input_dim = 3

        if cfg.post_inner_dim == "6dim":
            inner_dim = 6
        else:
            inner_dim = 3

        if cfg.postprocessor == "convolution":
            self.layer = ConvolutionPostprocessor(cfg, input_dim, inner_dim)

    def forward(self, param, trans):
        """
            input:
                param       (batch * duration * 312)
                trans       (batch * duration * 3)
            
            output:
                same
        """

        out_param, out_trans = self.layer(param, trans)
        return out_param, out_trans


class ConvolutionPostprocessor(nn.Module):
    def __init__(self, cfg, input_dim, inner_dim):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.conv1 = nn.Conv2d(1, inner_dim*52+3, (5, inner_dim*52+3), stride=1)


    def forward(self, param, trans):
        batch_size, input_duration = param.shape[:2]
        if self.input_dim != self.inner_dim:
            # 6dim -> euler
            param = param.reshape(batch_size, input_duration, 52, 6)
            param = rotation_6d_to_matrix(param)
            param = matrix_to_euler_angles(param, "ZYX")
            
            param = param.reshape(batch_size, input_duration, 156)

        pose = torch.cat((param, trans), dim=2)
        pose = nn.functional.pad(pose.unsqueeze(0), (0, 0, 2, 2), mode="replicate").squeeze(0).unsqueeze(1)
        pose = self.conv1(pose)
        
        # pose = pose.squeeze(1)
        pose = pose.squeeze(3).transpose(1, 2)

        out_param, out_trans = pose[:, :, :-3], pose[:, :, -3:]

        if self.input_dim != self.inner_dim:
            # euler -> 6dim
            out_param = out_param.reshape(batch_size, input_duration, 52, 3)
            out_param = euler_angles_to_matrix(out_param, "ZYX")
            out_param = matrix_to_rotation_6d(out_param)
            out_param = out_param.reshape(batch_size, input_duration, 312)

        return out_param, out_trans

