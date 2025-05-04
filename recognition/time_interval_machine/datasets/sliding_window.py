# import os,sys

# # 取得目前的工作目錄 (預期是 tim-main)
# current_dir = os.getcwd()
# print("Current Working Directory:", current_dir)

# # 設定 recognition 的路徑 (假設 recognition 位於 tim-main 底下)
# recognition_dir = os.path.join(current_dir, "recognition")
# os.chdir(recognition_dir)

# print("New Working Directory:", os.getcwd())

# sys.path.append(os.getcwd())
# print("sys.path:", sys.path)

import pandas as pd
import numpy as np
import torch
import math
import os

from torch.utils import data

import time_interval_machine.utils.logging as logging


logger = logging.get_logger(__name__)

def timestamp_to_seconds(timestamp):
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * 3600.00 + minutes * 60.0 + seconds
    return total_seconds

def load_feats(feat_info, data_path, mode):
    feat_times = {}
    feats = {}

    v_ids = feat_info['video_id'].unique().tolist()
    for v_id in v_ids:
        vid_times = feat_info[feat_info['video_id'] == v_id].sort_values('start_sec')
        vid_times = vid_times.drop(columns=['video_id', 'narration_sec'])
        feat_times[v_id] = torch.FloatTensor(vid_times.to_numpy())

        vid_feats = np.load(os.path.join(data_path, f'{mode}/{v_id}.npy'))
        if "slowfast" in str(data_path):
            vid_feats = np.expand_dims(vid_feats, axis=2) 
        feats[v_id] = torch.from_numpy(vid_feats)
        

    return feat_times, feats

class SlidingWindowDataset(data.Dataset):
    def __init__(self,
                v_data_path,
                a_data_path,
                v_action_labels_pickle,
                a_action_labels_pickle,
                v_context_labels_pickle,
                a_context_labels_pickle,
                video_info_pkl,
                v_feature_dim=1024,
                a_feature_dim=2304,
                num_feats=25,
                feat_stride=2,
                feat_gap=0.2,
                window_stride=1.0,
                mode='train',
                min_query_size=0.2,
                data_modality='audio_visual',
                model_modality='audio_visual',
                include_verb_noun=True,
                dataset_name='ave'
            ):

        logger.info("Constructing dataset for split : {}".format(mode))

        # Initialise parameters
        self.dataset_name = dataset_name
        self.mode = mode
        self.data_modality = data_modality
        self.model_modality = model_modality
        self.include_verb_noun = include_verb_noun

        self.v_feature_dim = v_feature_dim
        self.a_feature_dim = a_feature_dim

        self.num_feats = num_feats
        self.feat_stride = feat_stride
        self.feat_gap = feat_gap
        self.window_size = self.num_feats * feat_gap * feat_stride
        self.window_stride = window_stride
        self.min_query_size = min_query_size
        self.v_num_aug = 1
        self.a_num_aug = 1
        self.max_visual_actions = 0
        self.max_audio_actions = 0
        self.num_actions = 0
        self.min_query = 2*self.window_size
        self.max_query = 0
        self.avg_query = 0
        self.vis_mul = 3 if include_verb_noun else 1


        logger.info("Caching Features")
        self.cache_features(
                            v_data_path,
                            a_data_path,
                            v_context_labels_pickle,
                            a_context_labels_pickle
                        )

        logger.info("Creating Windows")
        self.init_windows(
                            v_action_labels_pickle,
                            a_action_labels_pickle,
                            video_info_pkl
                        )

        self.max_window_actions = self.max_visual_actions + self.max_audio_actions
        logger.info((f"{mode.capitalize()} Sliding Window dataset constructed. Total Actions: {self.num_actions}\n\
                    \t\t\t\tNumber of {self.window_size} Second Windows: {len(self.windows)}\n\
                    \t\t\t\tMax actions in window: {self.max_window_actions}\n\
                    \t\t\t\t\tVisual: {self.max_visual_actions}\n\
                    \t\t\t\t\tAudio: {self.max_audio_actions}\n\
                    \t\t\t\tMin Query Size: {self.min_query}\n\
                    \t\t\t\tMax Query Size: {self.max_query}\n\
                    \t\t\t\tAvg. Queries per Window: {self.avg_query}"))

    def cache_features(
                    self,
                    v_data_path,
                    a_data_path,
                    v_feat_times_pkl,
                    a_feat_times_pkl
                ):

        if "visual" in self.model_modality:
            logger.info("Loading visual data")
            feat_times = pd.read_pickle(v_feat_times_pkl)
            v_info = load_feats(feat_times, v_data_path, self.mode)
            self.v_feat_times, self.v_feats = v_info
            # print(self.v_feats["c---zaDCTaE"].shape)
            self.v_num_aug = list(self.v_feats.values())[0].size(1)
            # print(self.v_num_aug)
        else:
            self.v_feat_times = None
            self.v_feats = None
            self.v_num_aug = 0

        if "audio" in self.model_modality:
            logger.info("Loading audio data")
            feat_times = pd.read_pickle(a_feat_times_pkl)
            a_info = load_feats(feat_times, a_data_path, self.mode)
            
            self.a_feat_times, self.a_feats = a_info
            # print(self.a_feats["c---zaDCTaE"].shape)
            self.a_num_aug = list(self.a_feats.values())[0].size(1)
        else:
            self.a_feat_times = None
            self.a_feats = None
            self.a_num_aug = 0

    def init_windows(
                    self,
                    v_labels_pkl,
                    a_labels_pkl,
                    video_info_pkl
                ):
        keep_columns = [
                "video_id",
                "start_sec",
                "stop_sec",
                "verb_class",
                "noun_class",
                "action_class",
                "class_id"
            ]

        # Format visual and audio ground truth dataframes to a matching format
        if "visual" in self.data_modality:
            v_actions = pd.read_pickle(v_labels_pkl)
            v_actions["start_sec"] = v_actions["start_timestamp"].apply(timestamp_to_seconds)
            v_actions["stop_sec"] = v_actions["stop_timestamp"].apply(timestamp_to_seconds)
            if self.dataset_name == 'ave':
                v_actions["action_class"] = v_actions["class_id"]
            else:
                v_actions["class_id"] = [-1] * v_actions.shape[0]

            if "verb_class" not in v_actions.columns:
                v_actions["verb_class"] = [-1] * v_actions.shape[0]
                v_actions["noun_class"] = [-1] * v_actions.shape[0]

            v_actions = v_actions[keep_columns]
            v_actions.index = v_actions.index.set_names(['narration_id'])
            v_actions = v_actions.reset_index()
            v_actions['narration_id'] = v_actions['narration_id'].apply(lambda x: f"v_{x}")

        if "audio" in self.data_modality:
            a_actions = pd.read_pickle(a_labels_pkl)
            a_actions["start_sec"] = a_actions["start_timestamp"].apply(timestamp_to_seconds)
            a_actions["stop_sec"] = a_actions["stop_timestamp"].apply(timestamp_to_seconds)
            a_actions["verb_class"] = [-1] * a_actions.shape[0]
            a_actions["noun_class"] = [-1] * a_actions.shape[0]
            a_actions["action_class"] = [-1] * a_actions.shape[0]
            a_actions = a_actions[keep_columns]
            a_actions.index = a_actions.index.set_names(['narration_id'])
            a_actions = a_actions.reset_index()
            a_actions['narration_id'] = a_actions['narration_id'].apply(lambda x: f"a_{x}")

        if self.data_modality == "visual":
            actions = v_actions
        elif self.data_modality == "audio":
            actions = a_actions
        else:
            actions = pd.concat([v_actions, a_actions], axis=0)
            actions = actions.reset_index(drop=True)

        self.num_actions = actions.shape[0]

        video_info = pd.read_pickle(video_info_pkl)
        video_info = video_info[video_info.index.isin(actions['video_id'].unique())]

        all_n_ids = actions['narration_id'].tolist()
        actions = actions.groupby('video_id')
        # Create Windows
        self.windows = []
        num_queries = []
        windows_path = self.create_windows_path(v_labels_pkl, a_labels_pkl)


        # Check if windows are precomputed
        if not os.path.exists(windows_path):
            # Track if all actions are captured
            seen_actions = set([])
            for vid, data in video_info.iterrows():
                video_duration = math.ceil(data['duration'])
                num_windows_in_vid = max(math.ceil((math.ceil(video_duration) - self.window_size) / self.window_stride) + 1, 1)
                vid_actions = actions.get_group(vid).copy()

                # Some labels are longer than video info duration
                vid_actions['stop_sec'] = vid_actions['stop_sec'].apply(lambda x: min(x, video_duration))

                for w in range(num_windows_in_vid):
                    win_start = self.window_stride * w
                    win_stop = min(video_duration, win_start + self.window_size)
                    actions_in_segment = vid_actions[
                                ((vid_actions['start_sec'] < win_stop)
                                & (vid_actions['stop_sec'] > win_start))
                            ].copy()
                    if actions_in_segment.shape[0] > 0:
                        actions_in_segment['full_duration'] = round(actions_in_segment['stop_sec'] - actions_in_segment['start_sec'], 3)

                        # Calculate duration of any partial queries
                        actions_in_segment['start_sec'] = actions_in_segment['start_sec'].apply(lambda x: max(x, win_start))
                        actions_in_segment['stop_sec'] = actions_in_segment['stop_sec'].apply(lambda x: min(x, win_stop))
                        actions_in_segment['partial_duration'] = round(actions_in_segment['stop_sec'] - actions_in_segment['start_sec'], 3)

                        # Query all actions within the window
                        full_queries = ((actions_in_segment['partial_duration'] == actions_in_segment['full_duration']))

                        # Ensure any partial queries are of a minimum size
                        partial_queries = ((actions_in_segment['partial_duration'] >= self.min_query_size))

                        actions_in_segment = actions_in_segment[full_queries | partial_queries]
                        if actions_in_segment.shape[0] > 0:
                            vid_feat_times = self.v_feat_times[vid] if 'visual' in self.model_modality else self.a_feat_times[vid]
                            win_feats = self.get_window_features(vid_feat_times, win_start, win_stop)
                            self.min_query = min(self.min_query, min(actions_in_segment['partial_duration'].tolist()))
                            self.max_query = max(self.max_query, max(actions_in_segment['partial_duration'].tolist()))
                            actions_in_segment = actions_in_segment.drop(columns=['video_id', 'full_duration', 'partial_duration'])

                            action_times = np.array(actions_in_segment[['start_sec', 'stop_sec']])
                            action_times = torch.from_numpy(action_times).float()

                            action_labels = np.array(actions_in_segment[['verb_class', 'noun_class', 'action_class', 'class_id']])
                            action_labels = torch.from_numpy(action_labels).long()

                            a_ids = torch.LongTensor(actions_in_segment.index.tolist())
                            n_ids = actions_in_segment['narration_id'].tolist()

                            visual_actions = [i for (i, n_id) in enumerate(n_ids) if 'v_' in n_id]
                            audio_actions = [i for (i, n_id) in enumerate(n_ids) if 'a_' in n_id]

                            if len(visual_actions) > self.max_visual_actions:
                                self.max_visual_actions = actions_in_segment.shape[0]

                            if len(audio_actions) > self.max_audio_actions:
                                self.max_audio_actions = actions_in_segment.shape[0]

                            window_info = {
                                    'video_id': vid,
                                    'start_sec': win_start,
                                    'stop_sec': win_stop,
                                    'feat_indices': win_feats,
                                    'v_queries': action_times[visual_actions],
                                    'v_labels': action_labels[visual_actions],
                                    'v_action_ids': a_ids[visual_actions],
                                    'v_narration_ids': [n_ids[v] for v in visual_actions],
                                    'a_queries': action_times[audio_actions],
                                    'a_labels': action_labels[audio_actions],
                                    'a_action_ids': a_ids[audio_actions],
                                    'a_narration_ids': [n_ids[a] for a in audio_actions]
                                }

                            self.windows.append(window_info)

                            num_queries.append(actions_in_segment.shape[0])

                            seen_actions.update(set(n_ids))
            all_windows = {
                            "windows": self.windows,
                            "num_queries": num_queries,
                            "max_vis": self.max_visual_actions,
                            "max_aud": self.max_audio_actions,
                            "min_query": self.min_query,
                            "max_query": self.max_query,
                            "seen_actions": seen_actions
                        }
            torch.save(all_windows, windows_path)
        else:
            logger.info(f"Loading precomputed windows from {windows_path}")
            all_windows = torch.load(windows_path)
            self.windows = all_windows['windows']
            num_queries = all_windows['num_queries']
            self.max_visual_actions = all_windows['max_vis']
            self.max_audio_actions = all_windows['max_aud']
            self.min_query = all_windows['min_query']
            self.max_query = all_windows['max_query']
            seen_actions = all_windows['seen_actions']

        missing = set(all_n_ids).difference(seen_actions)
        assert len(missing) == 0, f"Windows only see {len(seen_actions)} / {self.num_actions}] actions. {missing}"
        self.avg_query = int(round(sum(num_queries) / len(num_queries)))

    def create_windows_path(self, v_labels_pkl, a_labels_pkl):
        windows_path = "precomputed_windows/"
        os.makedirs(windows_path, exist_ok=True)
        
        if "visual" in self.data_modality:
            if self.dataset_name not in str(v_labels_pkl).lower():
                windows_path += f"{self.dataset_name.upper()}_"
            v_label_name = str(v_labels_pkl).split('/')[-1].replace('.pkl', '')
            windows_path += f"{v_label_name}_"

            # AVE has same label set, so need to distinguish to avoid errors
            if self.dataset_name == 'ave':
                windows_path += f"visual_" 

        if "audio" in self.data_modality:
            if self.dataset_name not in str(a_labels_pkl).lower() and self.data_modality != "audio_visual":
                windows_path += f"{self.dataset_name.upper()}"
                
            a_labels_name = str(a_labels_pkl).split('/')[-1].replace('.pkl', '')
            windows_path += f"{a_labels_name}_"

            if self.dataset_name == 'ave':
                windows_path += f"audio_" 

        hop_secs = round(self.feat_stride * self.feat_gap, 3)
        windows_path += f"win_{self.num_feats}_{hop_secs}_{self.window_size}_{self.window_stride}.pth"
        return windows_path

    def __getitem__(self, index):
        window = self.windows[index]
        video_id = window['video_id']
        feat_indices = window['feat_indices']
        v_queries = window['v_queries']
        a_queries = window['a_queries']
        v_labels = window['v_labels']
        a_labels = window['a_labels']

        # Collect Times and Input Data
        times = torch.zeros(size=(0, 2))
        v_data = torch.empty(size=(0,))
        a_data = torch.empty(size=(0,))

        if "visual" in self.model_modality:
            v_aug_indices = torch.randint(
                                    low=0,
                                    high=self.v_num_aug,
                                    size=(self.num_feats,),
                                    dtype=torch.long
                                )
            v_data = self.v_feats[video_id][feat_indices, v_aug_indices]
            v_input_feat_times = self.v_feat_times[video_id][feat_indices, :2]
            times = torch.cat([times, v_input_feat_times], dim=0)

        if "audio" in self.model_modality:
            a_aug_indices = torch.randint(
                                    low=0,
                                    high=self.a_num_aug,
                                    size=(self.num_feats,),
                                    dtype=torch.long
                                )
            a_data = self.a_feats[video_id][feat_indices, a_aug_indices]
            a_input_feat_times = self.a_feat_times[video_id][feat_indices, :2]
            times = torch.cat([times, a_input_feat_times], dim=0)

        # Collect and pad queries+metadata
        v_to_pad = (0, 0 , 0, self.max_visual_actions - v_labels.size(0))
        v_queries = torch.nn.functional.pad(v_queries, v_to_pad, "constant", 0.0)
        v_labels = torch.nn.functional.pad(v_labels, v_to_pad, "constant", -1)
        v_action_ids = torch.nn.functional.pad(
                            window['v_action_ids'],
                            (0, v_to_pad[3]),
                            "constant",
                            -1
                        )
        v_narration_ids = window['v_narration_ids'] + [''] * v_to_pad[3]

        # Pad queries to max size and form times for Time MLP
        a_to_pad = (0, 0 , 0, self.max_audio_actions - a_labels.size(0))
        a_queries = torch.nn.functional.pad(a_queries, a_to_pad, "constant", 0.0)
        a_labels = torch.nn.functional.pad(a_labels, a_to_pad, "constant", -1)
        a_action_ids = torch.nn.functional.pad(
                            window['a_action_ids'],
                            (0, a_to_pad[3]),
                            "constant",
                            -1
                        )
        a_narration_ids = window['a_narration_ids'] + [''] * a_to_pad[3]

        # Normalize times and make relative to input size
        times = torch.concat([times, v_queries, a_queries], dim=0)
        times = (times - window['start_sec']) / self.window_size
        times = torch.clamp(times, min=0.0)

        label = {
                'verb': v_labels[:, 0],
                'noun': v_labels[:, 1],
                'action': v_labels[:, 2],
                'class_id': a_labels[:, 3]
            }
        metadata = {
                'v_action_ids': v_action_ids,
                'a_action_ids': a_action_ids,
                'v_narration_ids': v_narration_ids,
                'a_narration_ids': a_narration_ids,
                'num_v_queries': self.max_visual_actions,
                'num_a_queries': self.max_audio_actions
            }

        return v_data, a_data, times, label, metadata

    def __len__(self):
        return len(self.windows)

    def get_window_features(self, feat_times, window_start, window_stop):
        # Get features to represent windowed input
        start_time = max(0.0, window_start)
        input_start = np.absolute(feat_times[:, 0] - start_time).argmin()
        input_end = np.absolute(feat_times[:, 1] - window_stop).argmin()

        feat_indices = list(range(input_start, input_end, self.feat_stride))
        feat_indices = np.clip(feat_indices, 0, len(feat_times) - 1)
        feat_indices = torch.from_numpy(feat_indices).long()

        num_pad = self.num_feats - feat_indices.size(0)
        final_index = feat_indices[-1]
        feat_indices = torch.nn.functional.pad(feat_indices, (0, num_pad), "constant", final_index)

        return feat_indices


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # 設定檔案路徑，請依需求修改
    v_data_path = "/mnt/2Tkioxia/yu/feats_ominvoreB_npy"
    a_data_path = "/mnt/2Tkioxia/yu/feats_slowfast_npy"
    v_action_labels_pickle = "/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_train.pkl"
    v_context_labels_pickle = "/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_train_feature_times.pkl"
    a_action_labels_pickle = "/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_train.pkl"
    a_context_labels_pickle = "/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_train_feature_times.pkl"
    video_info_pkl = "/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_video_info.pkl"

    # 設定 model_modality 可選 "fused" 或 "audio_visual"
    dataset = SlidingWindowDataset(
        v_data_path=v_data_path,
        a_data_path=a_data_path,
        v_action_labels_pickle=v_action_labels_pickle,
        v_context_labels_pickle=v_context_labels_pickle,
        a_action_labels_pickle=a_action_labels_pickle,
        a_context_labels_pickle=a_context_labels_pickle,
        video_info_pkl=video_info_pkl,
        v_feature_dim=1024,
        a_feature_dim=2304,
        num_feats=25,
        feat_stride=2,
        feat_gap=0.2,
        window_stride=1.0,
        mode='train',
        min_query_size=0.2,
        dataset_name='ave',
        data_modality='audio_visual',
        model_modality='audio_visual',
        include_verb_noun=False,
    )
    
    batch_size = 16
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    # 示範印出前 batch_size 個視窗的資訊
    print("Batch 中每個視窗的查詢時間（原始秒數）、features 的時間、對應的 index 與每個 query 的 label：")
    for idx in range(batch_size):
        window = dataset.windows[idx]
        video_id = window['video_id']
        print(f"\nWindow {idx} (video_id: {video_id}):")
        
        queries = window['a_queries']
        # if queries.shape[0] != 4:
        #     print(f"  Window {idx} 的查詢數量不為 4，實際查詢數量為 {queries.shape[0]}")
        #     break
        print(" 查詢時間 (原始秒數):")
        for q_idx, query in enumerate(queries):
            start_time, end_time = query.tolist()
            print(f"  Query {q_idx}: start = {start_time}, end = {end_time}")
        
        labels = window['a_labels']
        print(" 查詢對應的 label (class_id):")
        for q_idx, lab in enumerate(labels.tolist()):
            print(f"  Query {q_idx}: label = {lab[3]}")
        
        feat_indices = window['feat_indices']
        print(" Features 對應的 index:")
        print(f"  {feat_indices.tolist()}")
        feat_times = dataset.a_feat_times[video_id][feat_indices, :2]
        print(" Features 時間 (原始秒數):")
        for f_idx, (f_start, f_end) in enumerate(feat_times.tolist()):
            print(f"  Feature_ {f_idx}: start = {f_start}, end = {f_end}")

    # 示範取得 DataLoader 的 batch 資料
    # for batch in dataloader:
    #     data_feats, times, labels, metadata = batch
    #     if dataset.model_modality == 'fused':
    #         print("data_feats shape (fused):", data_feats.shape)
    #     else:
    #         print("data_feats (包含 audio 與 visual) keys:", list(data_feats.keys()))
    #         print("audio shape:", data_feats['audio'].shape)
    #         print("visual shape:", data_feats['visual'].shape)
    #     print("times shape:", times.shape)
    #     print("labels['class_id'] shape:", labels['class_id'].shape)
    #     break
