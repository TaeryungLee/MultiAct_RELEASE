import torch
import numpy as np
import math
from torch.nn import functional as F
import torchgeometry as tgm
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, euler_angles_to_matrix, matrix_to_axis_angle, matrix_to_euler_angles, rotation_6d_to_matrix


def face_front_align(params, transl, target_rot_vec, target_transl):
    """
    Regularize motion by given axis angle rotation and translation
    inputs:
        params ((batch): optional * duration * num_param): global orientation in first three parameters
        transl ((batch): optional * duration * 3)
        target_rot_vec ((batch) * 3): rotation to be set as an orientation
        target_transl ((batch) * 3): translation to be set as an orientation
    """
    if type(params).__module__ == "numpy":
        params = torch.from_numpy(params).to(torch.float32)
    if type(transl).__module__ == "numpy":
        transl = torch.from_numpy(transl).to(torch.float32)
    if type(target_rot_vec).__module__ == "numpy":
        target_rot_vec = torch.from_numpy(target_rot_vec).to(torch.float32)
    if type(target_transl).__module__ == "numpy":
        target_transl = torch.from_numpy(target_transl).to(torch.float32)
    params = params.clone()
    transl = transl.clone()
    target_rot_vec = target_rot_vec.clone()
    target_transl = target_transl.clone()
    no_batch = False
    if len(params.shape) == 2:
        params.unsqueeze_(0)
        transl.unsqueeze_(0)
        target_rot_vec.unsqueeze_(0)
        target_transl.unsqueeze_(0)
        no_batch = True

    batch, duration = params.shape[:2]

    # basic global orient: look forward
    basic_global_orient = axis_angle_to_matrix(torch.tensor((0.00001, 0, 0))).expand(batch, 3, 3)

    # inverse target rotation 
    inv_target_rot_mat = axis_angle_to_matrix(target_rot_vec)  # batch * 3 * 3
    inv_target_rot_mat = matrix_to_euler_angles(inv_target_rot_mat, "ZYX")

    inv_target_rot_mat[ :, 1] = 0
    inv_target_rot_mat[ :, 2] = 0
    inv_target_rot_mat = euler_angles_to_matrix(inv_target_rot_mat, "ZYX")

    inv_target_rot_mat = inv_target_rot_mat.transpose(1, 2)  # rotation matrix is orthonormal, so transpose = inverse
    inv_target_rot_mat = torch.bmm(basic_global_orient, inv_target_rot_mat)

    inv_target_rot_mat = inv_target_rot_mat.unsqueeze(1).expand(batch, duration, 3, 3)  # batch * duration * 3 * 3

    orig_global_orient_mat = axis_angle_to_matrix(params[:, :, :3])  # batch * duration * 3 * 3
    new_global_orient_mat = torch.bmm(inv_target_rot_mat.reshape(-1, 3, 3), orig_global_orient_mat.reshape(-1, 3, 3))
    new_global_orient_mat = new_global_orient_mat.reshape(batch, duration, 3, 3)
    new_global_orient_vec = matrix_to_axis_angle(new_global_orient_mat)

    params[:, :, :3] = new_global_orient_vec

    transl = transl - target_transl
    transl = transl.reshape(batch*duration, 3, 1)
    transl = torch.bmm(inv_target_rot_mat.reshape(batch * duration, 3, 3), transl).reshape(batch, duration, 3)

    if no_batch:
        params.squeeze_(0)
        transl.squeeze_(0)

    return params, transl


def to_front_view(params):
    """
    Inputs:
        params (duration * 156)
    Outputs:
        params (duration * 156)
    """
    params = params.clone()
    global_rot = params[:, :3].clone()
    inv_global_rot = axis_angle_to_matrix(global_rot)
    inv_global_rot = matrix_to_euler_angles(inv_global_rot, "ZYX")
    inv_global_rot[:, 1] = 0
    inv_global_rot[:, 2] = 0
    inv_global_rot = euler_angles_to_matrix(inv_global_rot, "ZYX")

    inv_global_rot = inv_global_rot.transpose(1, 2)

    orig_global_rot = axis_angle_to_matrix(params[:, :3].clone())
    new_global_rot = torch.bmm(inv_global_rot, orig_global_rot)
    new_global_rot = matrix_to_axis_angle(new_global_rot)

    params[:, :3] = new_global_rot

    return params


def to_right_view(params):
    """
    Inputs:
        params (duration * 156)
    Outputs:
        params (duration * 156)
    """
    params = params.clone()
    duration = params.shape[0]
    global_rot = params[:, :3].clone()
    global_rot = axis_angle_to_matrix(global_rot)

    # right turn matrix
    right_rot = euler_angles_to_matrix(torch.tensor([math.pi/2, 0, 0]), "ZYX").unsqueeze(0).expand(duration, 3, 3)

    new_global_rot = torch.bmm(right_rot, global_rot)
    new_global_rot = matrix_to_axis_angle(new_global_rot)

    params[:, :3] = new_global_rot

    return params


def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix

    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle
