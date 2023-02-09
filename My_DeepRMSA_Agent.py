from __future__ import division
import os
import struct
import types
import string
import numpy as np
import math
import copy
import random
import datetime

import threading
import multiprocessing
import platform
import tensorflow as tf
import wandb
# import tensorflow.contrib.slim as slim
import tf_slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time
from collections import defaultdict

import pickle as pk
from AC_Net import AC_Net
from utils import *

tf.compat.v1.enable_eager_execution()


class DeepRMSAAgent(object):

    def __init__(self,
                 agent_id,
                 trainer,
                 linkmap,
                 LINK_NUM,
                 NODE_NUM,
                 SLOT_TOTAL,
                 k_path,
                 M,
                 lambda_req,
                 lambda_time,
                 len_lambda_time,
                 gamma,
                 episode_size,
                 batch_size,
                 Src_Dest_Pair,
                 Candidate_Paths,
                 num_src_dest_pair,
                 model_path,
                 global_episodes,
                 regu_scalar,
                 x_dim_p,
                 x_dim_v,
                 n_actions,
                 num_layers,
                 layer_size,
                 model2_flag,
                 nonuniform,
                 prob_arr,
                 results_path,
                 configfile,
                 maxDR=600):
        # tf.compat.v1.disable_v2_behavior()
        self.name = 'agent_' + str(agent_id)
        self.trainer = trainer
        self.linkmap = linkmap
        self.LINK_NUM = LINK_NUM
        self.NODE_NUM = NODE_NUM
        self.SLOT_TOTAL = SLOT_TOTAL
        self.k_path = k_path
        self.M = M
        self.lambda_req = lambda_req
        self.lambda_time = lambda_time
        self.len_lambda_time = len_lambda_time
        self.gamma = gamma
        self.episode_size = episode_size
        self.batch_size = batch_size
        self.Src_Dest_Pair = Src_Dest_Pair
        self.Candidate_Paths = Candidate_Paths
        self.num_src_dest_pair = num_src_dest_pair
        self.model_path = model_path
        self.model2_flag = model2_flag
        self.nonuniform = nonuniform
        self.prob_arr = prob_arr
        self.results_path = results_path
        self.maxDR = maxDR

        self.global_episodes = global_episodes  #
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_blocking = []
        self.episode_mean_values = []
        self.summary_writer = tf.compat.v1.summary.FileWriter("tmp/train_{}/{}".format(self.name,
                                                                                       datetime.datetime.now().strftime(
                                                                                           '%Y-%m-%d_%H-%M-%S')))

        self.x_dim_p = x_dim_p
        self.x_dim_v = x_dim_v
        self.n_actions = n_actions

        self.local_network = AC_Net(scope=self.name,
                                    trainer=self.trainer,
                                    x_dim_p=self.x_dim_p,
                                    x_dim_v=self.x_dim_v,
                                    n_actions=self.n_actions,
                                    num_layers=num_layers,
                                    layer_size=layer_size,
                                    regu_scalar=regu_scalar)
        self.update_local_ops = self.update_target_graph('global', self.name)
        #
        self.slot_map = [[1 for x in range(self.SLOT_TOTAL)] for y in
                         range(self.LINK_NUM)]  # Initialized to be all available
        self.slot_map_t = [[0 for x in range(self.SLOT_TOTAL)] for y in
                           range(self.LINK_NUM)]  # the time each FS will be occupied
        self.service_time = self.lambda_time[np.random.randint(0, self.len_lambda_time)]
        self.lambda_intervals = 1 / self.lambda_req  # average time interval between request
        self.request_set = {}
        self.his_slotmap = []
        # self.all_ones = [[1 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)
        # self.all_negones = [[0 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)
        self.configfile = configfile
        self.configs = readconfigs(self.configfile)
        self.equalPSD = self.configs['PSD'].mean() if self.configs['PSD'].max() / self.configs['PSD'].min() - 1 < .01 \
            else None
        if self.name == 'agent_0':
            self.results_path = self.init_results_dir()
            self.model_path = self.init_model_dir()
        # self.all_ones = [[1 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)
        # self.all_negones = [[0 for x in range(self.LINK_NUM)] for y in range(self.LINK_NUM)] # (flag-slicing)

    def init_results_dir(self):
        save_dir = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        path = os.path.join('tmp/results', save_dir)
        if os.path.isdir(path):
            assert 'Directory already exists'
        else:
            os.mkdir(path)
        with open(os.path.join(path, 'symparam.txt'), "w") as f:
            f.write('k path: ' + str(self.k_path))
            f.write('\nM: ' + str(self.M))
            f.write('\nconfig: ' + str(self.configfile))
        return path

    def init_model_dir(self):
        save_dir = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        path = os.path.join('tmp/model', save_dir)
        if os.path.isdir(path):
            assert 'Directory already exists'
        else:
            os.mkdir(path)
        return path

    def store(self, bp, ep_values, mean_value_loss, mean_policy_loss, mean_entropy):
        names = ['BP', 'value', 'value_loss', 'policy_loss', 'entropy']
        for i, val in enumerate((bp, np.mean(ep_values), float(mean_value_loss), float(mean_policy_loss),
                                 float(mean_entropy))):
            with open(os.path.join(self.results_path, '{}.dat'.format(names[i])), 'a') as fp:
                fp.write('%f\n' % val)
        return True

    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def _get_path(self, src, dst, candidate_Paths, k):  # get path k of from src->dst
        if src == dst:
            print('error: _get_path()')
            path = []
        else:
            path = candidate_Paths[src][dst][k]
            if path is None:
                return None
        return path

    def calclink(self, p):  # map path to links
        path_link = []
        for a, b in zip(p[:-1], p[1:]):
            k = self.linkmap[a][b][0]
            path_link.append(k)
        return path_link

    def get_new_slot_temp(self, slot_temp, path_link, slot_map):
        for i in path_link:
            for j in range(self.SLOT_TOTAL):
                slot_temp[j] = slot_map[i][j] & slot_temp[j]
        return slot_temp

    def mark_vector(self, vector, default):
        le = len(vector)
        flag = 0
        slotscontinue = []
        slotflag = []

        ii = 0
        while ii <= le - 1:
            tempvector = vector[ii:le]
            default_counts = tempvector.count(default)
            if default_counts == 0:
                break
            else:
                a = tempvector.index(default)
                ii += a
                flag += 1
                slotflag.append(ii)
                m = vector[ii + 1:le]
                m_counts = m.count(1 - default)
                if m_counts != 0:
                    n = m.index(1 - default)
                    slotcontinue = n + 1
                    slotscontinue.append(slotcontinue)
                    ii += slotcontinue
                else:
                    slotscontinue.append(le - ii)
                    break
        return flag, slotflag, slotscontinue

    def judge_availability(self, slot_temp, current_slots, FS_id):
        (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
        fs = -1
        fe = -1
        if flag > 0:
            n = len(slotscontinue)
            flag_availability = 0  # Initialized to be unavailable
            t = 0
            for i in range(n):
                if slotscontinue[i] >= current_slots:
                    if t == FS_id:
                        fs = slotflag[i]
                        fe = slotflag[i] + current_slots - 1
                        flag_availability = 1
                        return flag_availability, fs, fe
                    t += 1
            return flag_availability, fs, fe
        else:
            flag_availability = 0
        return flag_availability, fs, fe

    def update_slot_map_for_committing_wp(self, slot_map, current_wp_link, current_fs, current_fe, slot_map_t,
                                          current_TTL):  # update slotmap, mark allocated FS' as occupied
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
                assert slot_map[ll][s] == 1
                assert slot_map_t[ll][s] == 0
                slot_map[ll][s] = 0
                slot_map_t[ll][s] = current_TTL
        return slot_map, slot_map_t

    def update_slot_map_for_releasing_wp(self, slot_map, current_wp_link, current_fs,
                                         current_fe):  # update slotmap, mark released FS' as free
        for ll in current_wp_link:
            for s in range(current_fs, current_fe + 1):
                assert slot_map[ll][s] == 0
                slot_map[ll][s] = 1
        return slot_map

    def update_release_matrix(self, release_matrix, current_wp_link, fs, fe):
        for ll in current_wp_link:
            for s in range(fs, fe + 1):
                release_matrix[ll][s] += 1

        return release_matrix

    def release(self, slot_map, request_set, slot_map_t,
                time_to, release_matrix):  # update slotmap to release FS' occupied by expired requests
        if request_set:
            # update slot_map_t
            for ii in range(self.LINK_NUM):
                for jj in range(self.SLOT_TOTAL):
                    if slot_map_t[ii][jj] > time_to:
                        slot_map_t[ii][jj] -= time_to
                    elif slot_map_t[ii][jj] > 0:
                        slot_map_t[ii][jj] = 0
            #
            del_id = []
            for rr in request_set:
                request_set[rr][3] -= time_to  # request_set[rr][3] is TTL
                if request_set[rr][3] <= 0:
                    current_wp_link = request_set[rr][0]
                    fs_wp = request_set[rr][1]
                    fe_wp = request_set[rr][2]
                    # release slots on the working path of the request
                    slot_map = self.update_slot_map_for_releasing_wp(slot_map, current_wp_link, fs_wp, fe_wp)
                    release_matrix = self.update_release_matrix(release_matrix, current_wp_link, fs_wp, fe_wp)
                    del_id.append(rr)
            for ii in del_id:
                del request_set[ii]
        return slot_map, request_set, slot_map_t, release_matrix

    def cal_len(self, path):
        path_len = 0
        for a, b in zip(path[:-1], path[1:]):
            path_len += self.linkmap[a][b][1]
        return path_len

    # calculate linear SNR in dB for specified path
    def cal_snr(self, path, margin=0.5):
        span_len = 80
        planckc = 6.626068e-34
        c = 2.99792458e8
        wavelength = 1.55e-6
        NF = 5
        NF_lin = 10 ** (NF / 10)
        attenuation = 0.2
        ASElist = []
        ASE_PSD = 0
        for a, b in zip(path[:-1], path[1:]):
            link_len = self.linkmap[a][b][1]
            nspans = int(np.ceil(link_len / span_len))
            for idx in range(nspans - 1):
                span_loss_lin = 10 ** (span_len * attenuation / 10)
                spont_emission = NF_lin / 2 * span_loss_lin / (span_loss_lin - 1) - 1 / span_loss_lin  # Kumar [6.104]
                ASE_PSD += spont_emission * planckc * c / wavelength * (span_loss_lin - 1)  # [7.15] Becker EDFA Book
            # loss on remaining span
            span_loss_lin = 10 ** ((link_len - (nspans - 1) * span_len) * attenuation / 10)
            spont_emission = NF_lin / 2 * span_loss_lin / (span_loss_lin - 1) - 1 / span_loss_lin  # Kumar [6.104]
            ASE_PSD += spont_emission * planckc * c / wavelength * (span_loss_lin - 1)  # [7.15] Becker EDFA Book

        ASE_PSD = (2 * ASE_PSD * 1e9)
        ASElist.append(ASE_PSD)
        linSNR = (10 * np.log10(self.equalPSD / np.asarray(ASE_PSD)) - margin).tolist()
        # GSNR = 10 * np.log10(1 / (1 / 10 ** (LPDict[CUTID]['linSNR'] / 10) + 1 / nliSNR_lin))
        return linSNR

    # DeepRMSA standard FS calculation based on linear SNR and resulting modulation format
    def cal_FS(self, reqDR, path, path_len=None, slot_temp=None):
        provDR = 0
        # ks = True
        # Only slots valid that fulfill SNR requirements of the path
        valid_configs = self.configs[self.configs['reqSNR'] < self.cal_snr(path)].sort_values(
            # by=['DR', 'modf', 'slot', 'configID'], ascending=(False, True, True, True))
            by=['DR', 'slot', 'configID'], ascending=(False, True, True))
        considered_configs = valid_configs[valid_configs['DR'] >= reqDR]

        # Get free available slots in the spectrum
        # flag, slotflag, slotscontinue = self.mark_vector(slot_temp)

        if considered_configs.empty:
            current_config = valid_configs.iloc[0]
            provDR += current_config['DR']
            num_FS = np.ceil(current_config['slot'] / 12.5)

        else:
            current_config = considered_configs.sort_values(by=['slot', 'DR'], ascending=[True, True]).iloc[0]
            provDR += current_config['DR']
            num_FS = np.ceil(current_config['slot'] / 12.5)

        '''max_slots = max(slotscontinue)
        # If free available slots are not sufficient, allocate maximum possible and choose lower config
        if current_config['slot'] / 12.5 > max_slots:
            num_FS = max_slots

            considered_configs = valid_configs[valid_configs['slot'] <= num_FS * 12.5].sort_values(by=['DR', 'SR'],
                                                                                                   ascending=[False,
                                                                                                              False])
            # Demand blocked, if no config available
            if considered_configs.empty:
                num_FS = 0
                # DR Will be returned but not allocated
                provDR = 0
                ks = False

            else:
                current_config = considered_configs.iloc[0]
                provDR = current_config['DR']
                num_FS = np.ceil(current_config['slot'] / 12.5)
                ks = False'''

        return int(num_FS), current_config['configID'], provDR

    def count_occupied(self, tm):
        zeros = 0
        tm = np.asarray(tm)
        for j in range(self.LINK_NUM):
            zeros_link = np.count_nonzero(tm[j] == 0)
            zeros += zeros_link
        return zeros

    def count_rset_slots(self):
        # temp_ = [list(path_links), fs_start, fs_end, current_TTL]
        slots = 0
        if self.request_set and self.request_set.__len__() != 0:
            for rr in self.request_set:
                slots += len(self.request_set[rr][0]) * (self.request_set[rr][2] + 1 - self.request_set[rr][1])
        return slots

    # Discounting function used to calculate discounted returns.
    def discount(self, x):
        return scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]

    def train(self, espisode_buff, sess, value_est):
        espisode_buff = np.array(espisode_buff, dtype=object)
        input_p = espisode_buff[:self.batch_size, 0]
        input_v = espisode_buff[:self.batch_size, 1]
        actions = espisode_buff[:self.batch_size, 2]
        rewards = espisode_buff[:, 3]
        values = espisode_buff[:, 4]

        self.rewards_plus = np.asarray(rewards.tolist() + [value_est])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]
        discounted_rewards = np.append(discounted_rewards, 0)  # --
        discounted_rewards_batch = discounted_rewards[:self.batch_size] - (
                self.gamma ** self.batch_size) * discounted_rewards[self.batch_size:]  # --
        self.value_plus = np.asarray(values.tolist() + [value_est])
        '''advantages = rewards + self.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.discount(advantages)
        advantages = np.append(advantages, 0) # --
        advantages = advantages[:self.batch_size] - (self.gamma**self.batch_size)*advantages[self.batch_size:] # --'''
        advantages = discounted_rewards_batch - self.value_plus[:self.batch_size]

        # a filtering scheme, filter out 20% largest and smallest elements from 'discounted_rewards_batch'
        '''sorted_reward = np.argsort(discounted_rewards_batch)
        input_p = input_p[sorted_reward[10:-10]]
        input_v = input_v[sorted_reward[10:-10]]
        discounted_rewards_batch = discounted_rewards_batch[sorted_reward[10:-10]]
        actions = actions[sorted_reward[10:-10]]
        advantages = advantages[sorted_reward[10:-10]]'''

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            self.local_network.target_v: discounted_rewards_batch,
            self.local_network.Input_p: np.vstack(input_p),
            self.local_network.Input_v: np.vstack(input_v),
            self.local_network.actions: actions,
            self.local_network.advantages: advantages
        }

        sum_value_losss, sum_policy_loss, sum_entropy, grad_norms_policy, grad_norms_value, var_norms_policy, \
            var_norms_value, _, _, regu_loss_policy, regu_loss_value = sess.run(
                [
                    self.local_network.loss_value,
                    self.local_network.loss_policy,
                    self.local_network.entropy,
                    self.local_network.grad_norms_policy,
                    self.local_network.grad_norms_value,
                    self.local_network.var_norms_policy,
                    self.local_network.var_norms_value,
                    self.local_network.apply_grads_policy,
                    self.local_network.apply_grads_value,
                    self.local_network.regu_loss_policy,
                    self.local_network.regu_loss_value
                ], feed_dict=feed_dict
            )
        return sum_value_losss / self.batch_size, sum_policy_loss / self.batch_size, sum_entropy / self.batch_size, grad_norms_policy, grad_norms_value, var_norms_policy, var_norms_value, regu_loss_policy / self.batch_size, regu_loss_value / self.batch_size

    def rmsa(self, sess, coord, saver):

        max_slots_used = self.configs['slot'].max() / 12.5
        min_slots_used = self.configs['slot'].min() / 12.5
        range_slots = max_slots_used - min_slots_used
        time_to = 0
        req_id = 0

        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        episode_buffer = []

        action_onehot = [x for x in range(self.n_actions)]
        sd_onehot = [x for x in range(self.num_src_dest_pair)]

        node_onehot = np.diag([1 for x in range(self.NODE_NUM)]).tolist()

        all_zeros = [0 for ii in range(3 + 2 * self.M)]
        all_nega_ones = [-1 for ii in range(3 + 2 * self.M)]

        # update local dnn with the global one
        sess.run(self.update_local_ops)

        # Initialize exploration probability (very high in beginning?!)
        epsilon = wandb.config.epsilon

        # bool signifying if next request is remaining reqDR of previous one
        carry = False

        print('Starting ' + self.name)
        with sess.as_default(), sess.graph.as_default():
            # Stuff for writer
            step = tf.Variable(0, dtype=tf.int64)
            step_update = step.assign_add(1)
            all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
            # writer_flush = self.summary_writer.flush()
            while not coord.should_stop():

                # store some traffic details:
                epsilons = []
                ids = []
                srcs = []
                dsts = []
                bws = []
                toa = []
                tod = []
                block = []
                action_ids = []
                path_ids = []
                fs_ids = []
                provDRs = []
                slotnums = []
                path_lens = []
                psds = []
                snrs = []
                carries = []
                occupied = []
                traffic_matrixs = []
                chosen_configs = []
                req_set_lens = []
                req_set_slots = []

                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                actionss = []

                num_blocks = 0
                release_req_ids = []

                pairs = np.zeros((self.episode_size, 2))
                lens = []
                for i in range(self.episode_size):
                    temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]
                    lens.append(self.cal_len(self._get_path(temp[0], temp[1], self.Candidate_Paths, k=0)))
                    pairs[i, 0] = temp[0]
                    pairs[i, 1] = temp[1]
                ind_lens_sorted = np.argsort(np.asarray(lens), axis=0)[::-1]
                lens_sorted = np.asarray(lens)[ind_lens_sorted]
                sorted_DRs = (np.random.randint(50, self.maxDR, self.episode_size))
                sorted_pairs = pairs[ind_lens_sorted, :]
                # sorted_DRs[::-1].sort()

                # adaptive learning rate
                '''if episode_count % 10 == 0 and episode_count != 0:
                    self.local_network.trainer._lr = np.max([1e-6, self.local_network.trainer._lr - 1e-6])'''

                # begin an episode
                resource_util = []
                # while req_id < self.episode_size:

                release_matrix = np.zeros((self.LINK_NUM, self.SLOT_TOTAL), int)

                while episode_step_count < self.episode_size:

                    # generate current request
                    if not carry:
                        req_id += 1
                        ids.append(req_id)
                        (self.slot_map, self.request_set, self.slot_map_t, release_matrix) = self.release(self.slot_map,
                                                                                                          self.request_set,
                                                                                                          self.slot_map_t,
                                                                                                          time_to,
                                                                                                          release_matrix)

                        time_to = 0
                        while time_to == 0:
                            time_to = np.random.exponential(self.lambda_intervals)
                        if self.nonuniform is True:
                            sd_id = np.random.choice(sd_onehot, p=self.prob_arr)
                            # temp = self.Src_Dest_Pair[sd_id]

                        else:
                            temp = self.Src_Dest_Pair[np.random.randint(0, self.num_src_dest_pair)]
                            temp = sorted_pairs[episode_step_count, :]
                        current_src = temp[0]
                        current_dst = temp[1]
                        current_DR = sorted_DRs[episode_step_count]
                        current_TTL = 0
                        while current_TTL == 0 or current_TTL >= self.service_time * 2:
                            current_TTL = np.random.exponential(self.service_time)
                    else:
                        ids.append(req_id)
                        current_DR -= provDR
                        '''(self.slot_map, self.request_set, self.slot_map_t) = self.release(self.slot_map,
                                                                                          self.request_set,
                                                                                          self.slot_map_t, time_to)'''
                        carry = False

                    srcs.append(current_src)
                    dsts.append(current_dst)
                    bws.append(current_DR)
                    tod.append(current_TTL)
                    toa.append(time_to)

                    # generate features
                    '''TTL_norm = current_TTL/(2*self.service_time)'''
                    src_onehot = node_onehot[int(current_src) - 1]
                    dst_onehot = node_onehot[int(current_dst) - 1]
                    Input_feature = []
                    Input_feature += src_onehot  # s
                    Input_feature += dst_onehot  # d
                    '''Input_feature.append(TTL_norm)'''
                    # include features from each path
                    if self.model2_flag > 0:
                        slot_map_fur = []
                        slot_map_tmp = copy.deepcopy(self.slot_map)
                        request_set_tmp = copy.deepcopy(self.request_set)
                        slot_map_t_tmp = copy.deepcopy(self.slot_map_t)
                        for ii in range(self.model2_flag):
                            (slot_map_tmp, request_set_tmp,
                             slot_map_t_tmp, release_matrix) = self.release(slot_map_tmp,
                                                                            request_set_tmp,
                                                                            slot_map_t_tmp,
                                                                            5 * self.lambda_intervals,
                                                                            release_matrix)
                            slot_map_fur.append(slot_map_tmp)

                    for x in range(self.k_path):
                        path = self._get_path(current_src, current_dst, self.Candidate_Paths, x)
                        if len(path) == 0:
                            '''Input_feature += all_zeros'''
                            Input_feature += all_nega_ones
                            for ii in range(self.model2_flag):
                                Input_feature += [0, 0, 0]
                        else:
                            path_len = self.cal_len(path)  # physical length of the path
                            slot_temp = [1] * self.SLOT_TOTAL
                            path_links = self.calclink(path)
                            slot_temp = self.get_new_slot_temp(slot_temp, path_links,
                                                               self.slot_map)  # spectrum utilization on the whole path

                            num_FS, configID, provDR = self.cal_FS(current_DR, path, path_len, slot_temp)
                            '''if current_DR - provDR > 0:
                                carry = True
                            else:
                                carry = False'''

                            (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
                            if flag == 0 or np.max(slotscontinue) < num_FS:
                                '''Input_feature += all_zeros'''
                                Input_feature += all_nega_ones
                            else:
                                '''Input_feature.append((num_FS-1)/8) # number of FS's required using this path, 
                                2 <= num_FS <= 9 '''
                                '''Input_feature.append((num_FS - self.configs['slot'].min() / 12.5) / (
                                        self.configs['slot'].max() / 12.5 - self.configs['slot'].min() / 12.5))'''  # normalized
                                Input_feature.append((2 * (num_FS - min_slots_used) - (range_slots)) / (range_slots))
                                slotscontinue_array = np.array(slotscontinue)
                                idx = np.where(slotscontinue_array >= num_FS)[0]
                                for jj in range(self.M):  # for the first self.M available FS-blocks
                                    if len(idx) > jj:
                                        '''Input_feature.append(slotflag[idx[jj]]/self.SLOT_TOTAL) # starting index
                                        Input_feature.append(slotscontinue[idx[jj]]/8) # size
                                        Input_feature.append(
                                            2 * (slotflag[idx[jj]] - 0.5 * self.SLOT_TOTAL) / self.SLOT_TOTAL)'''
                                        Input_feature.append(slotflag[idx[jj]] / self.SLOT_TOTAL)
                                        '''Input_feature.append(2*(slotscontinue[idx[jj]] - 0.5*self.SLOT_TOTAL) / self.SLOT_TOTAL)'''
                                        Input_feature.append((slotscontinue[idx[jj]]) / self.SLOT_TOTAL)
                                    else:
                                        '''Input_feature += [0, 0]'''
                                        Input_feature += [-1, -1]
                                Input_feature.append(sum(slotscontinue) / len(slotscontinue) / self.SLOT_TOTAL)  # total
                                # available FS's
                                Input_feature.append((np.mean(
                                    slotscontinue) - self.SLOT_TOTAL / 2) / self.SLOT_TOTAL * 2)  # mean size of FS-blocks
                            # --------------------------------------------------
                            for ii in range(self.model2_flag):
                                slot_temp = [1] * self.SLOT_TOTAL
                                slot_temp = self.get_new_slot_temp(slot_temp, path_links, slot_map_fur[ii])
                                (flag, slotflag, slotscontinue) = self.mark_vector(slot_temp, 1)
                                if flag == 0:
                                    Input_feature += [0, 0, 0]
                                else:
                                    Input_feature.append(sum(slotscontinue) / self.SLOT_TOTAL)
                                    Input_feature.append(np.mean(slotscontinue) / max_slots_used)
                                    Input_feature.append(np.max(slotscontinue) / max_slots_used)
                            # --------------------------------------------------

                    Input_feature = np.array(Input_feature)
                    Input_feature = np.reshape(np.array(Input_feature), (1, self.x_dim_p))

                    blocking = 0

                    # Take an action using probabilities from policy network output.
                    prob_dist, value, entro = sess.run(
                        [self.local_network.policy, self.local_network.value, self.local_network.entropy],
                        feed_dict={
                            self.local_network.Input_p: Input_feature,
                            self.local_network.Input_v: Input_feature
                        }
                    )
                    pp = prob_dist[0]
                    assert not np.isnan(entro)
                    '''if self.name == 'agent_0':
                        print(pp, '--')'''

                    if random.random() < epsilon:
                        action_id = np.random.choice(action_onehot, p=pp)
                    else:
                        action_id = np.argmax(pp)
                    action_ids.append(action_id)

                    # shift action id to avoid that in most cases, action_id = 0 is the best
                    '''if current_src <= 2:
                        action_id_transf = action_id
                    elif current_src <= 4:
                        action_id_transf = (action_id + 1) % self.n_actions # cyclic right move, step 1
                    else:
                        action_id_transf = (action_id + 2) % self.n_actions # cyclic right move, step 2
                    path_id = action_id_transf // self.M  # path to use
                    FS_id = math.fmod(action_id_transf, self.M)'''

                    path_id = action_id // self.M  # path to use
                    path_ids.append(path_id)
                    FS_id = math.fmod(action_id, self.M)
                    fs_ids.append(FS_id)
                    path = self._get_path(current_src, current_dst, self.Candidate_Paths, path_id)

                    actionss.append(action_id)
                    psds.append(self.equalPSD)

                    # apply the selected action
                    if len(path) == 0:  # selected an invalid action
                        blocking = 1
                    else:
                        path_len = self.cal_len(path)  # physical length of the path
                        path_lens.append(path_len)
                        snrs.append(self.cal_snr(path))
                        num_FS, configID, provDR = self.cal_FS(current_DR, path, path_len, slot_temp)
                        if current_DR - provDR > 0:
                            carry = True
                        else:
                            carry = False

                        chosen_configs.append(configID)
                        provDRs.append(provDR)
                        slotnums.append(num_FS)
                        carries.append(carry)

                        slot_temp = [1] * self.SLOT_TOTAL
                        path_links = self.calclink(path)
                        slot_temp = self.get_new_slot_temp(slot_temp, path_links,
                                                           self.slot_map)  # spectrum utilization on the whole path
                        (flag, fs_start, fs_end) = self.judge_availability(slot_temp, num_FS, FS_id)
                        if flag == 1:
                            self.slot_map, self.slot_map_t = self.update_slot_map_for_committing_wp(
                                self.slot_map,
                                path_links,
                                fs_start,
                                fs_end,
                                self.slot_map_t,
                                current_TTL
                            )  # --
                            # update slotmap
                            temp_ = [list(path_links), fs_start, fs_end, current_TTL]  # update in-service requests
                            self.request_set[episode_step_count] = temp_  # request_set[req_id] maybe carry overwrites

                        else:
                            blocking = 1
                            # if FS_id > 0:
                            #     blocking += 49

                    req_set_lens.append(len(self.request_set))
                    r_t = 1 - 2 * blocking  # successful, 1, blocked, -1
                    # r_t = 1 - blocking
                    block.append(blocking)
                    num_blocks += blocking
                    occupied.append(self.count_occupied(self.slot_map))
                    req_set_slots.append(self.count_rset_slots())

                    episode_reward += r_t
                    total_steps += 1
                    episode_step_count += 1

                    resource_util.append(1 - np.sum(self.slot_map) / (self.LINK_NUM * self.SLOT_TOTAL))

                    if episode_count < (3000 / self.episode_size):  # for warm-up
                        continue

                    # store experience
                    episode_buffer.append([Input_feature, Input_feature, action_id, r_t, value[0, 0]])
                    episode_values.append(value[0, 0])

                    if len(episode_buffer) == 2 * self.batch_size - 1:
                        mean_value_losss, mean_policy_loss, mean_entropy, grad_norms_policy, grad_norms_value, \
                            var_norms_policy, var_norms_value, regu_loss_policy, \
                            regu_loss_value = self.train(episode_buffer, sess, 0.0)
                        del (episode_buffer[:self.batch_size])
                        sess.run(
                            self.update_local_ops)  # if we want to synchronize local with global every time a
                        # training is performed
                        epsilon = np.max([epsilon - 1e-5, 0.05])
                        epsilons.append(epsilon)

                # traffic_matrixs.append(self.slot_map)
                storage_interval = 100
                if episode_count % storage_interval == 0 and self.name == 'agent_0':
                    # Store environment variables and metrics
                    df = pd.DataFrame(
                        (epsilons, episode_values, ids, srcs, dsts, bws, provDRs, psds, snrs, toa, tod, block,
                         action_ids, path_ids, fs_ids, slotnums, path_lens, chosen_configs, carries, req_set_lens,
                         occupied, req_set_slots)
                    ).T
                    df.columns = (
                        'epsilons', 'values', 'IDs', 'srcNode', 'dstNode', 'reqDR', 'provDR', 'PSD', 'linSNR', 'toa',
                        'tod', 'block', 'Action ID', 'Path ID', 'FS ID', 'Slots', 'Path length', 'configID',
                        'carry', 'rset lengths', 'zero slots', 'rset slots')
                    det_filename = 'net_details_' + str(int(episode_count // storage_interval)) + '.csv'
                    traffic_matrix_filename = 'traffic_matrix_' + str(int(episode_count // storage_interval)) + '.csv'
                    traffic_matrix_t_filename = 'traffic_matrix_t_' + str(
                        int(episode_count // storage_interval)) + '.csv'
                    release_matrix_filename = 'release_matrix_' + str(int(episode_count // storage_interval)) + '.csv'
                    df.to_csv(os.path.join(self.results_path, det_filename))

                    df = pd.DataFrame(self.slot_map,
                                      columns=np.arange(191.0 * 1e3, 191 * 1e3 + 100 * 12.5, 12.5))
                    df.to_csv(os.path.join(self.results_path, traffic_matrix_filename))

                    df = pd.DataFrame(self.slot_map_t,
                                      columns=np.arange(191.0 * 1e3, 191 * 1e3 + 100 * 12.5, 12.5))
                    df.to_csv(os.path.join(self.results_path, traffic_matrix_t_filename))

                    df = pd.DataFrame(release_matrix,
                                      columns=np.arange(191.0 * 1e3, 191 * 1e3 + 100 * 12.5, 12.5))
                    df.to_csv(os.path.join(self.results_path, release_matrix_filename))

                    # if platform.processor() == 'x86_64':
                    #    os.

                    print('stored successfully!')

                    # if episode_count > 0:
                    #     coord.request_stop()

                # end of an episode
                episode_count += 1

                # sess.run(self.update_local_ops) # if we want to synchronize local with global every episode is
                # finished

                if episode_count <= (3000 / self.episode_size):  # for warm-up
                    continue

                bp = num_blocks / self.episode_size
                # bp = num_blocks / list(self.request_set.keys())[-1]
                if self.name == 'agent_0':
                    print('Episode Count = ', episode_count)
                    print('Blocking Probability = ', bp)
                    # print('Action Distribution', actionss.count(0)/len(actionss))
                    print('Mean Resource Utilization =', np.mean(resource_util))
                    if self.store(bp, episode_values, mean_value_losss, mean_policy_loss, mean_entropy):
                        print('----')

                self.episode_blocking.append(bp)
                self.episode_rewards.append(episode_reward)
                self.episode_mean_values.append(np.mean(episode_values))

                # Periodically save model parameters, and summary statistics.
                sample_step = int(1000 / self.episode_size)
                if episode_count % sample_step == 0 and episode_count != 0:
                    if episode_count % (100 * sample_step) == 0 and self.name == 'agent_0':
                        saver.save(sess, self.model_path + '/model.cptk')
                        # Save all files that currently exist containing the substring "ckpt":
                        wandb.save('../logs/*ckpt*')
                        # Save any files starting with "checkpoint" as they're written to:
                        wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
                        print("Model Saved")

                    if self.name == 'agent_0':
                        mean_reward = np.mean(self.episode_rewards[-sample_step:])
                        mean_value = np.mean(self.episode_mean_values[-sample_step:])
                        mean_blocking = np.mean(self.episode_blocking[-sample_step:])
                        wandb.log(
                            {
                                'epsilon': epsilon,
                                'BP': bp,
                                'Perf/Reward': float(mean_reward),
                                'Perf/Value': float(mean_value),
                                'Perf/Blocking': float(mean_blocking),
                                'Losses/Value Loss': float(mean_value_losss),
                                'Losses/Policy Loss': float(mean_policy_loss),
                                'Losses/Entropy': float(mean_entropy),
                                'Losses/Grad Norm Policy': float(grad_norms_policy),
                                'Losses/Grad Norm Value': float(grad_norms_value),
                                'Losses/Var Norm Policy': float(var_norms_policy),
                                'Losses/Var Norm Value': float(var_norms_value),
                                'Losses/Regu Loss Policy': float(regu_loss_policy),
                                'Losses/Regu Loss Value': float(regu_loss_value)
                            }
                        )
                        wandb.log(
                            {
                                'Slot Map (Value Function)': wandb.plots.HeatMap(
                                    list(range(self.SLOT_TOTAL)),
                                    list(range(self.LINK_NUM)),
                                    self.slot_map
                                )
                            }
                        )

                        wandb.tensorflow.log(tf.compat.v1.summary.merge_all())
                        # self.summary_writer.add_summary(summary, episode_count)

                if self.name == 'agent_0':
                    sess.run(self.increment)

                ''' # End
                if (episode_count // 100) == 8:  # 4
                    wandb.finish()
                    coord.request_stop()'''
