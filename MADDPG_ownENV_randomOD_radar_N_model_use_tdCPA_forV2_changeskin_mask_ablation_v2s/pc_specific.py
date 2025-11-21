# -*- coding: utf-8 -*-
"""
@Time    : 11/6/2025 4:18 pm
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""


class PCconfig:
    def __init__(self):
        self.save_file_prefix = 'D:\MADDPG_2nd_jp/'
        self.shape_path = '/home/bizhao/lakesideMap'
        self.head_not_inside_mask = r'/home/bizhao/MADDPG_ownENV_randomOD_radar_N_model_use_tdCPA_forV2_changeskin_mask_ablation/mask_model_out_mask_wTanh_v2.pth'
        self.head_all_mask_head = r'/home/bizhao/MADDPG_ownENV_randomOD_radar_N_model_use_tdCPA_forV2_changeskin_mask_ablation/mask_model_fully_masked_wTanh_v2.pth'
        self.head_inside_mask = r'/home/bizhao/MADDPG_ownENV_randomOD_radar_N_model_use_tdCPA_forV2_changeskin_mask_ablation/mask_model_eps20000_in_mask_wTanh_v2.pth'