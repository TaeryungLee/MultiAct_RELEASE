import os
from utils.smplx.body_model.body_model import BodyModel

class SMPLH(object):
    def __init__(self, cfg):
        self.layer_arg = {
            'create_body_pose': False, 'create_betas': True, 'create_global_orient': False, 'create_transl': False,
            'create_left_hand_pose': False, 'create_right_hand_pose': False, 'ext': 'npz'
        }
        # self.layer = smplx.create(cfg.human_model_path, 'smplh', gender='MALE', **self.layer_arg)
        # self.layer = smplx.create("/mnt/ssd1/data/human_models/smplh", 'smplh', gender='MALE', **self.layer_arg)
        self.layer = BodyModel(
            bm_fname=os.path.join(cfg.human_model_path, "smplh", "SMPLH_MALE.pkl"), num_betas=10
        )

        self.joint_regressor = self.layer.J_regressor
        # self.orig_hand_regressor = {
        #     'left': self.layer.J_regressor.numpy()[[20, 37, 38, 39, 25, 26, 27, 28, 29, 30, 34, 35, 36, 31, 32, 33], :],
        #     'right': self.layer.J_regressor.numpy()[[21, 52, 53, 54, 40, 41, 42, 43, 44, 45, 49, 50, 51, 46, 47, 48],
        #              :]}

        self.face = self.layer.f
        self.shape_param_dim = 10

        # original SMPLH joint set
        self.orig_joint_num = 52  # 22 (body joints) + 30 (hand joints)
        self.orig_joints_name = \
            ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Foot',
             'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
             'L_Wrist', 'R_Wrist',  # body joints
             'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2', 'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2',
             'L_Pinky_3', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',  # left hand joints
             'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2', 'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2',
             'R_Pinky_3', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
             # right hand joints
             )
        self.orig_flip_pairs = \
            ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21),  # body joints
             (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42), (28, 43), (29, 44), (30, 45), (31, 46),
             (32, 47), (33, 48), (34, 49), (35, 50), (36, 51)  # hand joints
             )
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = \
            {'body': range(self.orig_joints_name.index('Pelvis'), self.orig_joints_name.index('R_Wrist') + 1),
             'lhand': range(self.orig_joints_name.index('L_Index_1'), self.orig_joints_name.index('L_Thumb_3') + 1),
             'rhand': range(self.orig_joints_name.index('R_Index_1'), self.orig_joints_name.index('R_Thumb_3') + 1)}

        self.root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.lwrist_idx = self.orig_joints_name.index('L_Wrist')
        self.rwrist_idx = self.orig_joints_name.index('R_Wrist')
        self.neck_idx = self.orig_joints_name.index('Neck')
