import copy
import gc
import json
import re
import os

from ldm.modules.diffusionmodules.openaimodel import Downsample, ResBlock, UNetModel, Upsample, timestep_embedding
from modules import sd_models, shared, devices
# from scripts.mbw_util.preset_weights import PresetWeights
import torch
from natsort import natsorted
from easing_functions import *

import gradio as gr
import modules.images as webui_modules_images
import modules.scripts as scripts
import torch
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from modules import script_callbacks
from modules import script_callbacks, sd_hijack_clip, sd_hijack_open_clip
from modules.processing import (Processed, StableDiffusionProcessing, fix_seed,
                                process_images)
from modules.shared import cmd_opts, opts, state
from PIL import Image
from scripts.daam import trace, utils

# noinspection DuplicatedCode
easing_function_namelist = ['LinearInOut',
                            'QuadEaseInOut',
                            'QuadEaseIn',
                            'QuadEaseOut',
                            'CubicEaseInOut',
                            'CubicEaseIn',
                            'CubicEaseOut',
                            'QuarticEaseInOut',
                            'QuarticEaseIn',
                            'QuarticEaseOut',
                            'QuinticEaseInOut',
                            'QuinticEaseIn',
                            'QuinticEaseOut',
                            'SineEaseInOut',
                            'SineEaseIn',
                            'SineEaseOut',
                            'CircularEaseIn',
                            'CircularEaseInOut',
                            'CircularEaseOut',
                            'ExponentialEaseInOut',
                            'ExponentialEaseIn',
                            'ExponentialEaseOut',
                            'ElasticEaseIn',
                            'ElasticEaseInOut',
                            'ElasticEaseOut',
                            'BackEaseIn',
                            'BackEaseInOut',
                            'BackEaseOut',
                            'BounceEaseIn',
                            'BounceEaseInOut',
                            'BounceEaseOut',
                            ]

easing_function_list = [lambda x: x,
                        QuadEaseInOut(),
                        QuadEaseIn(),
                        QuadEaseOut(),
                        CubicEaseInOut(),
                        CubicEaseIn(),
                        CubicEaseOut(),
                        QuarticEaseInOut(),
                        QuarticEaseIn(),
                        QuarticEaseOut(),
                        QuinticEaseInOut(),
                        QuinticEaseIn(),
                        QuinticEaseOut(),
                        SineEaseInOut(),
                        SineEaseIn(),
                        SineEaseOut(),
                        CircularEaseIn(),
                        CircularEaseInOut(),
                        CircularEaseOut(),
                        ExponentialEaseInOut(),
                        ExponentialEaseIn(),
                        ExponentialEaseOut(),
                        ElasticEaseIn(),
                        ElasticEaseInOut(),
                        ElasticEaseOut(),
                        BackEaseIn(),
                        BackEaseInOut(),
                        BackEaseOut(),
                        BounceEaseIn(),
                        BounceEaseInOut(),
                        BounceEaseOut(),
                        ]

# presetWeights = PresetWeights()


shared.UNetBManager = None
shared.UnetAdapter = None


def cat_modded(tensors, *args, **kwargs):
    if len(tensors) == 2:
        a, b = tensors
        if a.shape[-2:] != b.shape[-2:]:
            a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

        tensors = (a, b)

    return torch.cat(tensors, *args, **kwargs)


# noinspection DuplicatedCode
class DebugUNetAdapter(object):
    def __init__(self, org_module: torch.nn.Module = None):
        super().__init__()

        self.org_module = org_module
        self.org_forward = None
        self.updown_block_adapters = []
        self.step_records = []
        self.temp_states = []
        self.layer_weight_list = []
        self.current_layer_idx = 0
        self.enable_weight_control = False
        self.enable_analysis = False
        self.delta_analysis = False
        self.cycle_counter = -1
        self.shadowing = False
        self.shadowX = None
        self.shadowXdownFeed = None

        self.last_x = None

    def __str__(self):
        return "Debug " + str(self.org_module)

    def set_unet(self, org_module: torch.nn.Module):
        self.org_module = org_module

    def check_unet(self):
        return self.org_module is not None

    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward']:
            return getattr(self.org_module, attr)

    def clear_states(self):

        for step_state_list in self.step_records:
            for state_variable in step_state_list:
                del state_variable
        del self.step_records
        for state_variable in self.temp_states:
            del state_variable
        del self.temp_states
        del self.shadowX
        del self.shadowXdownFeed

        self.step_records = []

        self.temp_states = []

        self.current_layer_idx = 0
        self.cycle_counter = -1
        self.shadowX = None
        self.shadowXdownFeed = None

    def get_final_results(self):
        # self.step_records.append(self.temp_states)
        return self.step_records

    def get_block_weight_info_suite(self):
        if self.enable_weight_control:
            return self.layer_weight_list[self.current_layer_idx], self.last_x
        else:
            return 1, self.last_x

    def get_enable_weight_control(self):
        return self.enable_weight_control

    def add_temp_state(self, new_state):
        self.temp_states.append(new_state)
        print(len(self.temp_states))

    def set_analysis_delta(self, delta_flag):
        self.delta_analysis = delta_flag

    def set_last_x(self, new_last_x):
        self.last_x = new_last_x

    def set_shadowing(self, new_shadowing_flag):
        self.shadowing = new_shadowing_flag

    def get_shadowing(self):
        return self.shadowing

    def set_enable_analysis(self, new_analysis_flag):
        self.enable_analysis = new_analysis_flag

    def get_enable_analysis(self):
        return self.enable_analysis

    def get_delta_analysis_flag(self):
        return self.delta_analysis

    def set_weight(self, layer_weight_list):
        unique_weight_value_set = set(layer_weight_list)
        if (layer_weight_list is not None and len(layer_weight_list) == 25 and
                (len(unique_weight_value_set) > 1 or 1 not in unique_weight_value_set)):
            self.enable_weight_control = True
            self.layer_weight_list = layer_weight_list
        else:
            self.enable_weight_control = False
            self.layer_weight_list = []

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        if timesteps[0] >= 999:
            # only analyzing batch 1
            self.cycle_counter += 1
        # if shared.newStart:
        # shared.UNetBManager.model_state_apply_all_blocks(timesteps[0] / 1000)
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = cat_modded([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

    def apply_to(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward


# noinspection PyMethodMayBeStatic,DuplicatedCode
class UNetStateManager(object):
    def __init__(self, org_unet: UNetModel):
        super().__init__()
        self.modelC_state_dict = None
        self.modelC_state_dict_by_blocks = []
        self.device = devices.device
        self.modelB_state_dict_by_blocks = []
        self.torch_unet = org_unet
        # self.modelA_state_dict = copy.deepcopy(org_unet.state_dict())
        self.modelA_state_dict = None
        self.dtype = devices.dtype
        self.modelA_state_dict_by_blocks = []
        # self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
        self.modelB_state_dict = None
        self.unet_block_module_list = []
        self.unet_block_module_list = [*self.torch_unet.input_blocks, self.torch_unet.middle_block, self.torch_unet.out,
                                       *self.torch_unet.output_blocks, self.torch_unet.time_embed]
        self.applied_weights = [0] * 27
        # self.gui_weights = [0.5] * 27
        self.enabled = False
        self.modelA_path = shared.sd_model.sd_model_checkpoint
        self.modelB_path = ''
        self.modelC_path = ''
        self.reversed = False
        self.easing_function = lambda x: x
        self.merge_target = float(1)

        self.intpoint_datas = []
        self.intpoint_intervals = []
        self.intpoint_targets = []

    # def set_gui_weights(self, current_weights):
    #     self.gui_weights = current_weights

    def reload_model_a(self):

        if self.modelA_path == shared.sd_model.sd_model_checkpoint:
            return
        self.modelA_path = shared.sd_model.sd_model_checkpoint
        if not self.enabled:
            return
        del self.modelA_state_dict_by_blocks
        self.modelA_state_dict_by_blocks = []
        # orig_modelA_state_dict_keys = list(self.modelA_state_dict.keys())
        # for key in orig_modelA_state_dict_keys:
        #     del self.modelA_state_dict[key]
        del self.modelA_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        self.modelA_state_dict = {k: v.cpu().to(self.dtype) for k, v in self.torch_unet.state_dict().items()}
        self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
        # if self.enabled:
        # self.model_state_apply(self.gui_weights)
        # self.model_state_apply(self.applied_weights)
        print('model A reloaded')

    def load_model_b(self, model_b_path):
        model_info = sd_models.get_closet_checkpoint_match(model_b_path)
        checkpoint_file = model_info.filename
        self.modelB_path = checkpoint_file
        if self.modelA_path == checkpoint_file:
            if not self.modelB_state_dict:
                self.enabled = False
            # self.gui_weights = current_weights
            return False
        # move initialization of model A to here
        if not self.modelA_state_dict:
            self.modelA_state_dict = {k: v.cpu().to(self.dtype) for k, v in self.torch_unet.state_dict().items()}
            # self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
        sd_model_hash = model_info.hash
        cache_enabled = shared.opts.sd_checkpoint_cache > 0

        if cache_enabled and model_info in sd_models.checkpoints_loaded:
            # use checkpoint cache
            print(f"Loading weights [{sd_model_hash}] from cache")
            self.modelB_state_dict = sd_models.checkpoints_loaded[model_info]
        device = "cpu"
        # if not self.device:
        #     self.device = self.torch_unet.device
        if self.modelB_state_dict:
            # orig_modelB_state_dict_keys = list(self.modelB_state_dict.keys())
            # for key in orig_modelB_state_dict_keys:
            #     del self.modelB_state_dict[key]
            del self.modelB_state_dict_by_blocks
            del self.modelB_state_dict
            torch.cuda.empty_cache()
            gc.collect()
        self.modelB_state_dict_by_blocks = []
        self.modelB_state_dict = self.filter_unet_state_dict(
            sd_models.read_state_dict(checkpoint_file, map_location=device))
        if len(self.modelA_state_dict) != len(self.modelB_state_dict):
            print('modelA and modelB state dict have different length, aborting')
            return False
        # self.map_blocks(self.modelB_state_dict, self.modelB_state_dict_by_blocks)
        # verify self.modelA_state_dict and self.modelB_state_dict have same structure
        # self.model_state_apply(current_weights)

        print('model B loaded')
        self.enabled = True
        return True

    # def load_modelC(self, modelC_path):
    #     model_info = sd_models.get_closet_checkpoint_match(modelC_path)
    #     checkpoint_file = model_info.filename
    #     self.modelC_path = checkpoint_file
    #     if self.modelA_path == checkpoint_file:
    #         if not self.modelC_state_dict:
    #             self.enabled = False
    #         # self.gui_weights = current_weights
    #         return False
    #     # move initialization of model A to here
    #     if not self.modelA_state_dict:
    #         self.modelA_state_dict = {k: v.cpu().to(self.dtype) for k, v in self.torch_unet.state_dict().items()}
    #         # self.map_blocks(self.modelA_state_dict, self.modelA_state_dict_by_blocks)
    #     sd_model_hash = model_info.hash
    #     cache_enabled = shared.opts.sd_checkpoint_cache > 0
    #
    #     if cache_enabled and model_info in sd_models.checkpoints_loaded:
    #         # use checkpoint cache
    #         print(f"Loading weights [{sd_model_hash}] from cache")
    #         self.modelC_state_dict = sd_models.checkpoints_loaded[model_info]
    #     device = "cpu"
    #     # if not self.device:
    #     #     self.device = self.torch_unet.device
    #     if self.modelC_state_dict:
    #         # orig_modelC_state_dict_keys = list(self.modelC_state_dict.keys())
    #         # for key in orig_modelC_state_dict_keys:
    #         #     del self.modelC_state_dict[key]
    #         del self.modelC_state_dict_by_blocks
    #         del self.modelC_state_dict
    #         gc.collect()
    #         torch.cuda.empty_cache()
    #     self.modelC_state_dict_by_blocks = []
    #     self.modelC_state_dict = self.filter_unet_state_dict(
    #         sd_models.read_state_dict(checkpoint_file, map_location=device))
    #     if len(self.modelA_state_dict) != len(self.modelC_state_dict):
    #         print('modelA and modelC state dict have different length, aborting')
    #         return False
    #     # self.map_blocks(self.modelC_state_dict, self.modelC_state_dict_by_blocks)
    #     # verify self.modelA_state_dict and self.modelC_state_dict have same structure
    #     # self.model_state_apply(current_weights)
    #
    #     print('model C loaded')
    #     self.enabled = True
    #     return True

    # def init_intpoints(self):
    #     intpoint_timepoints = []
    #     for int_target_suite in self.intpoint_targets:
    #         timepoint = int_target_suite[0]
    #         targets = int_target_suite[1:]
    #
    #         curpointDict = {}
    #         for weight_key in self.modelA_state_dict:
    #             curpointDict[weight_key] = (targets[0] * self.modelA_state_dict[weight_key] + targets[1] * self.modelB_state_dict[weight_key] + targets[2] * self.modelC_state_dict[weight_key]).to(device=self.device)
    #
    #         self.intpoint_datas.append(curpointDict)
    #         intpoint_timepoints.append(timepoint)
    #     # convert intpoint_times to interval pairs
    #     for i in range(len(intpoint_timepoints) - 1):
    #         self.intpoint_intervals.append((intpoint_timepoints[i], intpoint_timepoints[i + 1]))
    #
    #     print('intpoints initialized')

    # def model_state_apply(self, current_weights):
    #     # self.gui_weights = current_weights
    #     for i in range(27):
    #         cur_block_state_dict = {}
    #         for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
    #             if shared.cmd_opts.lowvram:
    #                 try:
    #                     curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
    #                                                  self.modelB_state_dict_by_blocks[i][cur_layer_key].to(self.dtype),
    #                                                  current_weights[i])
    #                 except RuntimeError as e:
    #                     self.modelA_state_dict_by_blocks[i][cur_layer_key] = self.modelA_state_dict_by_blocks[i][
    #                         cur_layer_key].to('cpu')
    #                     curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
    #                                                  self.modelB_state_dict_by_blocks[i][cur_layer_key].to(self.dtype),
    #                                                current_weights[i])
    #             else:
    #                 curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key], self.modelB_state_dict_by_blocks[i][cur_layer_key].to(self.dtype), current_weights[i])
    #             cur_block_state_dict[cur_layer_key] = curlayer_tensor
    #         self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
    #     self.applied_weights = current_weights

    # def model_state_apply_modified_blocks(self, current_weights, current_model_B):
    #     if not self.enabled:
    #         return
    #     modelB_info = sd_models.get_closet_checkpoint_match(current_model_B)
    #     checkpoint_file_B = modelB_info.filename
    #     if checkpoint_file_B != self.modelB_path:
    #         print('model B changed, shouldn\'t happenm ')
    #         return
    #     if self.applied_weights == current_weights:
    #         return
    #     for i in range(27):
    #         if current_weights[i] != self.applied_weights[i]:
    #             cur_block_state_dict = {}
    #             for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
    #                 curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key], self.modelB_state_dict_by_blocks[i][cur_layer_key].to(self.dtype), current_weights[i])
    #                 cur_block_state_dict[cur_layer_key] = curlayer_tensor
    #             self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
    #     self.applied_weights = current_weights

    def model_state_apply_modified_blocks(self, current_weights, current_model_b):
        if not self.enabled:
            return
        model_b_info = sd_models.get_closet_checkpoint_match(current_model_b)
        checkpoint_file_b = model_b_info.filename
        if checkpoint_file_b != self.modelB_path:
            print('model B changed, shouldn\'t happen ')
            return
        if self.applied_weights == current_weights:
            return
        for i in range(27):
            if current_weights[i] != self.applied_weights[i]:
                cur_block_state_dict = {}
                for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key].to(self.dtype),
                                                 current_weights[i])
                    cur_block_state_dict[cur_layer_key] = curlayer_tensor
                self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights

    def model_state_apply_all_blocks(self, weight):
        if not self.enabled:
            return
        if not self.reversed:
            weight = 1.0 - weight
            weight = self.easing_function(weight)
            if self.merge_target != 1.0:
                weight = weight * self.merge_target
        else:
            weight_temp = 1.0 - weight
            weight_temp = self.easing_function(weight_temp)
            if self.merge_target != 1.0:
                weight_temp = weight_temp * self.merge_target
            weight = 1.0 - weight_temp
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, device=self.device)
        weight = weight.to(self.dtype)

        # convert to intpopint position
        interval_index = 0
        curinterval_progress = 0
        for index, intervalpair in enumerate(self.intpoint_intervals):

            if intervalpair[0] <= weight <= intervalpair[1]:
                interval_index = index
                curinterval_progress = (weight - intervalpair[0]) / (intervalpair[1] - intervalpair[0])
                break

        # print(weight)
        # if self.applied_weights == [weight] * 27:
        #     return
        cur_intpoint_start = self.intpoint_datas[interval_index]
        cur_intpoint_end = self.intpoint_datas[interval_index + 1]

        # for i in range(27):
        #
        #     cur_block_state_dict = {}
        #     for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
        #         if shared.cmd_opts.lowvram:
        #             weight = weight.to('cpu')
        #         curlayer_tensor = torch.lerp(cur_intpoint_start,
        #                                      cur_intpoint_end,
        #                                      curinterval_progress)
        #         cur_block_state_dict[cur_layer_key] = curlayer_tensor
        #     self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        # self.applied_weights = [weight] * 27
        cur_temp_state_dict = {}
        for weight_key in cur_intpoint_start:
            curlayer_tensor = torch.lerp(cur_intpoint_start[weight_key],
                                         cur_intpoint_end[weight_key],
                                         curinterval_progress)
            cur_temp_state_dict[weight_key] = curlayer_tensor
        self.torch_unet.load_state_dict(cur_temp_state_dict)

    def clean_intpoints(self):

        self.intpoint_intervals = []
        self.intpoint_targets = []
        # for idx in range(len(self.intpoint_datas)):
        #     data_keys = list(self.intpoint_datas[idx].keys())
        #     for key in data_keys:
        #         del self.intpoint_datas[idx][key]
        del self.intpoint_datas
        self.intpoint_datas = []
        torch.cuda.empty_cache()

    def model_a_reset(self):
        if self.applied_weights == [0] * 27:
            return
        for i in range(27):
            self.unet_block_module_list[i].load_state_dict(self.modelA_state_dict_by_blocks[i])
        self.applied_weights = [0] * 27

    # diff current_weights and self.applied_weights, apply only the difference
    def model_state_apply_block(self, current_weights):
        # self.gui_weights = current_weights
        if not self.enabled:
            return self.applied_weights
        for i in range(27):
            if current_weights[i] != self.applied_weights[i]:
                cur_block_state_dict = {}
                for cur_layer_key in self.modelA_state_dict_by_blocks[i]:
                    curlayer_tensor = torch.lerp(self.modelA_state_dict_by_blocks[i][cur_layer_key],
                                                 self.modelB_state_dict_by_blocks[i][cur_layer_key], current_weights[i])
                    cur_block_state_dict[cur_layer_key] = curlayer_tensor
                self.unet_block_module_list[i].load_state_dict(cur_block_state_dict)
        self.applied_weights = current_weights
        return self.applied_weights

    # filter input_dict to include only keys starting with 'model.diffusion_model'
    # def filter_unet_state_dict(self, input_dict):
    #     filtered_dict = {}
    #     for key, value in input_dict.items():
    #
    #         if key.startswith('model.diffusion_model'):
    #             filtered_dict[key[22:]] = value
    #     filtered_dict_keys = natsorted(filtered_dict.keys())
    #     filtered_dict = {k: filtered_dict[k] for k in filtered_dict_keys}
    #
    #     return filtered_dict

    def filter_unet_state_dict(self, input_dict):
        filtered_dict = {}
        for key, value in input_dict.items():

            if key.startswith('model.diffusion_model'):
                filtered_dict[key[22:]] = value.to(self.dtype)
        filtered_dict_keys = natsorted(filtered_dict.keys())
        filtered_dict = {k: filtered_dict[k] for k in filtered_dict_keys}

        return filtered_dict

    def map_blocks(self, model_state_dict_input, model_state_dict_by_blocks):
        if model_state_dict_by_blocks:
            print('mapping to non empty list')
            return
        model_state_dict_sorted_keys = natsorted(model_state_dict_input.keys())
        # sort model_state_dict by model_state_dict_sorted_keys
        model_state_dict = {k: model_state_dict_input[k] for k in model_state_dict_sorted_keys}

        known_block_prefixes = [
            'input_blocks.0.',
            'input_blocks.1.',
            'input_blocks.2.',
            'input_blocks.3.',
            'input_blocks.4.',
            'input_blocks.5.',
            'input_blocks.6.',
            'input_blocks.7.',
            'input_blocks.8.',
            'input_blocks.9.',
            'input_blocks.10.',
            'input_blocks.11.',
            'middle_block.',
            'out.',
            'output_blocks.0.',
            'output_blocks.1.',
            'output_blocks.2.',
            'output_blocks.3.',
            'output_blocks.4.',
            'output_blocks.5.',
            'output_blocks.6.',
            'output_blocks.7.',
            'output_blocks.8.',
            'output_blocks.9.',
            'output_blocks.10.',
            'output_blocks.11.',
            'time_embed.'
        ]
        current_block_index = 0
        processing_block_dict = {}
        for key in model_state_dict:
            # print(key)
            if not key.startswith(known_block_prefixes[current_block_index]):
                if not key.startswith(known_block_prefixes[current_block_index + 1]):
                    print(
                        f"unknown key {key} in statedict after block {known_block_prefixes[current_block_index]}, possible UNet structure deviation"
                    )
                    continue
                else:
                    model_state_dict_by_blocks.append(processing_block_dict)
                    processing_block_dict = {}
                    current_block_index += 1
            block_local_key = key[len(known_block_prefixes[current_block_index]):]
            processing_block_dict[block_local_key] = model_state_dict[key]

        model_state_dict_by_blocks.append(processing_block_dict)
        print('mapping complete')
        return


before_image_saved_handler = None


class Script(scripts.Script):
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"

    def __init__(self) -> None:
        super().__init__()
        # attention_texts: str,
        # hide_images: bool,
        # dont_save_images: bool,
        # hide_caption: bool,
        # use_grid: bool,
        # grid_layouyt: str,
        # alpha: float,
        # heatmap_image_scale: float

        self.enabled = None
        self.tracer = None
        self.grid_layouyt = None
        self.prompt_analyzer = None
        self.attentions = None
        self.images = list()
        self.attention_texts: str = ""
        self.hide_images: bool = False
        self.dont_save_images: bool = False
        self.hide_caption: bool = False
        self.use_grid: bool = False
        self.grid_layout: str = self.GRID_LAYOUT_AUTO
        self.alpha: float = 0.5
        self.heatmap_image_scale: float = 1.0
        self.heatmap_images = list()

        # if shared.UNetBManager is None:
        #     shared.UNetBManager = UNetStateManager(shared.sd_model.model.diffusion_model)
        #     from modules.call_queue import wrap_queued_call
        #
        #     def reload_modelA_checkpoint():
        #         if shared.opts.sd_model_checkpoint == shared.sd_model.sd_checkpoint_info.title:
        #             return
        #         sd_models.reload_model_weights()
        #         shared.UNetBManager.reload_modelA()
        #
        #     shared.opts.onchange("sd_model_checkpoint",
        #                          wrap_queued_call(reload_modelA_checkpoint), call=False)

    # noinspection DuplicatedCode
    def before_image_saved(self, params: script_callbacks.ImageSaveParams):
        batch_pos = -1
        if params.p.batch_size > 1:
            # regex matching is no longer working, as Batch size and Batch pos from textinfo are removed
            # https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/00dab8f10defbbda579a1bc89c8d4e972c58a20d
            # match = re.search(r"Batch pos: (\d+)", params.pnginfo['parameters'])
            # if match:
            #     batch_pos = int(match.group(1))
            if params.batch_pos is not None:
                batch_pos = params.batch_pos

        else:
            batch_pos = 0

        if batch_pos < 0:
            return

        if self.tracer is not None and len(self.attentions) > 0:
            with torch.no_grad():
                styled_prompot = shared.prompt_styles.apply_styles_to_prompt(params.p.prompt, params.p.styles)
                global_heat_map = self.tracer.compute_global_heat_map(self.prompt_analyzer, styled_prompot, batch_pos)

                if global_heat_map is not None:
                    heatmap_images = []
                    for attention in self.attentions:

                        img_size = params.image.size
                        caption = attention if not self.hide_caption else None

                        heat_map = global_heat_map.compute_word_heat_map(attention)
                        if heat_map is None:
                            print(f"No heatmaps for '{attention}'")

                        heat_map_img = utils.expand_image(heat_map, img_size[1],
                                                          img_size[0]) if heat_map is not None else None
                        img: Image.Image = utils.image_overlay_heat_map(params.image, heat_map_img, alpha=self.alpha,
                                                                        caption=caption,
                                                                        image_scale=self.heatmap_image_scale)

                        fullfn_without_extension, extension = os.path.splitext(params.filename)
                        full_filename = fullfn_without_extension + "_" + attention + extension

                        if self.use_grid:
                            heatmap_images.append(img)
                        else:
                            heatmap_images.append(img)
                            if not self.dont_save_images:
                                img.save(full_filename)

                    self.heatmap_images += heatmap_images

        # if it is last batch pos, clear heatmaps
        if batch_pos == params.p.batch_size - 1:
            self.tracer.reset()

        return

    # noinspection DuplicatedCode
    def process(self,
                p: StableDiffusionProcessing,
                *args):

        # assert opts.samples_save, "Cannot run Daam script. Enable 'Always save all generated images' setting."

        # attention_texts: str,
        # hide_images: bool,
        # dont_save_images: bool,
        # hide_caption: bool,
        # use_grid: bool,
        # grid_layout: str,
        # alpha: float,
        # heatmap_image_scale: float
        self.enabled = False  # in case assert fails
        initial_info = None

        self.images = []
        attention_texts = args[0]

        self.hide_images = args[1]
        self.dont_save_images = args[2]
        self.hide_caption = args[3]
        self.use_grid = args[4]
        self.grid_layout = args[5]
        self.alpha = args[6]

        self.heatmap_image_scale = args[7]
        self.heatmap_images = list()

        self.attentions = [s.strip() for s in attention_texts.split(",") if s.strip()]
        self.enabled = len(self.attentions) > 0

        # fix_seed(p)
        #
        # styled_prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)

    # noinspection PyMethodOverriding,DuplicatedCode
    def process_batch(self,
                      p: StableDiffusionProcessing,
                      *args,
                      **kwargs):

        if not self.enabled:
            return
        prompts = kwargs['prompts']
        styled_prompt = prompts[0]

        embedder = None
        if type(p.sd_model.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords or \
                type(p.sd_model.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
            embedder = p.sd_model.cond_stage_model
        else:
            assert False, f"Embedder '{type(p.sd_model.cond_stage_model)}' is not supported."

        tokenize = None

        if type(p.sd_model.cond_stage_model.wrapped) == FrozenCLIPEmbedder:
            clip: FrozenCLIPEmbedder = p.sd_model.cond_stage_model.wrapped
            tokenize = clip.tokenizer.tokenize
        elif type(p.sd_model.cond_stage_model.wrapped) == FrozenOpenCLIPEmbedder:
            clip: FrozenOpenCLIPEmbedder = p.sd_model.cond_stage_model.wrapped
            # noinspection PyProtectedMember
            tokenize = open_clip.tokenizer._tokenizer.encode
        else:
            assert False

        tokens = tokenize(utils.escape_prompt(styled_prompt))
        context_size = utils.calc_context_size(len(tokens))

        prompt_analyzer = utils.PromptAnalyzer(embedder, styled_prompt)
        self.prompt_analyzer = prompt_analyzer
        context_size = prompt_analyzer.context_size

        print(f"daam run with context_size={prompt_analyzer.context_size}, token_count={prompt_analyzer.token_count}")
        # print(f"remade_tokens={prompt_analyzer.tokens}, multipliers={prompt_analyzer.multipliers}")
        # print(f"hijack_comments={prompt_analyzer.hijack_comments}, used_custom_terms={prompt_analyzer.used_custom_terms}")
        # print(f"fixes={prompt_analyzer.fixes}")

        if any(item[0] in self.attentions for item in self.prompt_analyzer.used_custom_terms):
            print("Embedding heatmap cannot be shown.")

        global before_image_saved_handler

        def before_image_saved_handler_func(params):
            self.before_image_saved(params)

        before_image_saved_handler = before_image_saved_handler_func

        self.tracer = trace(p.sd_model, p.height, p.width, context_size).hook()



        # processed = Processed(p, self.images, p.seed, initial_info)

    # noinspection DuplicatedCode
    def postprocess(self, p, processed, *args, **kwargs):
        if not self.enabled:
            return

        if self.tracer is not None:
            self.tracer.unhook()
            self.tracer = None

        initial_info = None

        if initial_info is None:
            initial_info = processed.info

        self.images += processed.images

        global before_image_saved_handler
        before_image_saved_handler = None
        if len(self.heatmap_images) > 0:

            if self.use_grid:

                grid_layout = self.grid_layouyt
                if grid_layout == Script.GRID_LAYOUT_AUTO:
                    if p.batch_size * p.n_iter == 1:
                        grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                    else:
                        grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW

                if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                    grid_img = webui_modules_images.image_grid(self.heatmap_images)
                elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                    grid_img = webui_modules_images.image_grid(self.heatmap_images, batch_size=p.batch_size,
                                                               rows=p.batch_size * p.n_iter)
                else:
                    grid_img = webui_modules_images.image_grid(self.heatmap_images)

                if not self.dont_save_images:
                    webui_modules_images.save_image(grid_img, p.outpath_grids, "grid_daam", grid=True, p=p)

                if not self.hide_images:
                    processed.images.insert(0, grid_img)
                    processed.index_of_first_image += 1
                    processed.infotexts.insert(0, processed.infotexts[0])

            else:
                if not self.hide_images:
                    processed.images[:0] = self.heatmap_images
                    processed.index_of_first_image += len(self.heatmap_images)
                    processed.infotexts[:0] = [processed.infotexts[0]] * len(self.heatmap_images)

        return processed

    def title(self):
        return "Runtime block ensembling for UNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # noinspection DuplicatedCode
    def ui(self, is_img2img):
        with gr.Accordion('Runtime AGE', open=False):
            attention_texts = gr.Text(label='Attention texts for visualization. (comma separated)', value='')

            with gr.Row():
                hide_images = gr.Checkbox(label='Hide heatmap images', value=False)

                dont_save_images = gr.Checkbox(label='Do not save heatmap images', value=False)

                hide_caption = gr.Checkbox(label='Hide caption', value=False)

            with gr.Row():
                use_grid = gr.Checkbox(label='Use grid (output to grid dir)', value=False)

                grid_layout = gr.Dropdown(
                    [Script.GRID_LAYOUT_AUTO, Script.GRID_LAYOUT_PREVENT_EMPTY, Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW],
                    label="Grid layout",
                    value=Script.GRID_LAYOUT_AUTO
                )

            with gr.Row():
                alpha = gr.Slider(label='Heatmap blend alpha', value=0.5, minimum=0, maximum=1, step=0.01)

                heatmap_image_scale = gr.Slider(label='Heatmap image scale', value=1.0, minimum=0.1, maximum=1, step=0.025)

        return [attention_texts, hide_images, dont_save_images, hide_caption, use_grid, grid_layout, alpha,
                heatmap_image_scale]


def handle_before_image_saved(params: script_callbacks.ImageSaveParams):
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        before_image_saved_handler(params)

    return


script_callbacks.on_before_image_saved(handle_before_image_saved)
