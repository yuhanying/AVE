import numpy as np
import tqdm
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description=('Create .npy files for the AVE dataset'))

###### Things you need to modify ######
parser.add_argument('--feature_list', type=str, help='Path to extracted features',default="/mnt/2Tkioxia/yu/feats_ominvore/train/features/features.npy")
parser.add_argument('--metafile_list', type=str, help='Path to feature time intervals',default="/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_train_feature_times.pkl")
parser.add_argument('--out_dir', type=str, help='Path to save features to',default="/mnt/2Tkioxia/yu/feats_ominvoreB_w_tok_npy/train")
#######################################

def main(args):
    result_dict = {}
    print("Parsing the features")
    feature = np.load(args.feature_list,)
    metadata = pd.read_pickle(args.metafile_list)
    metadata = metadata.reset_index()
    for i in tqdm.tqdm(range(len(metadata))):
        annotation_id = metadata.iloc[i]['narration_id']
        vid_id = metadata.iloc[i]['video_id']
        if vid_id not in result_dict:
            result_dict[vid_id] = {}
        annotation_index = int(annotation_id.split('_')[-1])
        new_annotation_id = '{}_{:06d}'.format(vid_id, annotation_index)
        if new_annotation_id not in result_dict[vid_id]:
            result_dict[vid_id][new_annotation_id] = []
        
        result_dict[vid_id][new_annotation_id].append(feature[i])

    # SAVE npyfiles
    print("Saving as npy files")

    for vid_id, v_features in tqdm.tqdm(result_dict.items()):
        args.feature_list = []
        annotation_ids = sorted(list(v_features.keys()))
        for annotation_id in annotation_ids:
            feature = np.stack(v_features[annotation_id], axis=0)
            
            args.feature_list.append(feature)
        out_file = os.path.join(args.out_dir, f'{vid_id}.npy')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        vid_feature = np.stack(args.feature_list, axis=0)
        if len(vid_feature.shape) == 5:
            vid_feature = np.squeeze(vid_feature, axis=1)
            vid_feature = vid_feature.transpose(0, 1, 3, 2)
        # np.save(out_file, vid_feature)

        # SAVE npyfiles
    # print("Saving as npy files")

    # for vid_id, v_features in tqdm.tqdm(result_dict.items()):
    #     # 重用一個暫存 list 收集同一支影片所有 annotation 的 feature
    #     tmp_list = []
    #     annotation_ids = sorted(v_features.keys())
    #     for annotation_id in annotation_ids:
    #         feature = np.stack(v_features[annotation_id], axis=0)  # (num_clips, 1, 1, 1024, 16)
    #         tmp_list.append(feature)

    #     # 把這支影片所有 annotation 堆起來
    #     vid_feature = np.stack(tmp_list, axis=0)  # (45, 1, 1, 1024, 16) 假設有45個 annotation

    #     # 如果是5D (B,1,1,C,D) → 先把 dim=2 擠掉，再調換最後兩維
    #     if vid_feature.ndim == 5:
    #         # (B,1,1,1024,16) → (B,1,1024,16)
    #         vid_feature = np.squeeze(vid_feature, axis=2)
    #         # (B,1,1024,16) → (B,1,16,1024)
    #         vid_feature = vid_feature.transpose(0, 1, 3, 2)
    #     # 如果是4D (B,1,C,D) → 沿 axis=1 擠掉（舊版本的行為）
    #     elif vid_feature.ndim == 4:
    #         vid_feature = np.squeeze(vid_feature, axis=1)

        # 儲存
        out_file = os.path.join(args.out_dir, f'{vid_id}.npy')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.save(out_file, vid_feature)


if __name__ == "__main__":
    main(parser.parse_args())

#!/usr/bin/env python3
# """
# Convert a large .dat feature file (numpy memmap) into per-video .npy files
# using metadata intervals. Features are loaded lazily without full RAM usage.
# """
# import os
# import argparse
# import numpy as np
# import pandas as pd
# from tqdm import tqdm




# def main(args):
#     # Parse feature memmap parameters
#     num_videos = args.num_videos
#     num_clips = args.num_clips
#     feat_shape = tuple(args.feat_shape)  # e.g. (1024, 16, 7, 7)

#     # Open memmap for features: shape = (num_videos, num_clips, *feat_shape)
#     feature_mmap = np.memmap(
#         args.feature_list,
#         dtype='float32',
#         mode='r',
#         shape=(num_videos, num_clips, *feat_shape)
#     )

#     # Flatten first two dims => (num_videos*num_clips, *feat_shape)
#     feature = feature_mmap

#     # Load metadata DataFrame
#     metadata = pd.read_pickle(args.metafile_list)
#     metadata = metadata.reset_index()

#     # Group features per video and annotation
#     result_dict = {}
#     for i in tqdm(range(len(metadata)), desc="Grouping features"):
#         ann_id = metadata.loc[i, 'narration_id']
#         vid_id = metadata.loc[i, 'video_id']
#         # Initialize dicts
#         if vid_id not in result_dict:
#             result_dict[vid_id] = {}
#         # New annotation key
#         ann_idx = int(ann_id.split('_')[-1])
#         new_ann_id = f"{vid_id}_{ann_idx:06d}"
#         if new_ann_id not in result_dict[vid_id]:
#             result_dict[vid_id][new_ann_id] = []
#         # Append the feature vector/array
#         result_dict[vid_id][new_ann_id].append(feature[i])

#     # Save per-video .npy files
#     os.makedirs(args.out_dir, exist_ok=True)
#     for vid_id, ann_feats in tqdm(result_dict.items(), desc="Saving .npy per video"):
#         # For each annotation inside video
#         ann_ids = sorted(ann_feats.keys())
#         stacks = []
#         for ann in ann_ids:
#             # Stack all clips for this annotation
#             arr = np.stack(ann_feats[ann], axis=0)
#             stacks.append(arr)
#         # Concatenate along new axis: (num_annotations, num_clips, ...)
#         vid_feature = np.stack(stacks, axis=0)
#         # If features have a singleton dimension, squeeze
#         if vid_feature.ndim == 4:
#             vid_feature = np.squeeze(vid_feature, axis=1)
#         out_path = os.path.join(args.out_dir, f"{vid_id}.npy")
#         np.save(out_path, vid_feature)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Create per-video .npy from .dat memmap features and metadata."
#     )
#     parser.add_argument(
#         '--feature_list', type=str, default="/mnt/2Tkioxia/yu/feats_ominvore/val/features/features.npy",
#         help='Path to feature .dat file (numpy memmap)'
#     )
#     parser.add_argument(
#         '--num_videos', type=int,default=18045,
#         help='Total number of videos in the .dat memmap'
#     )
#     parser.add_argument(
#         '--num_clips', type=int,default=1,
#         help='Number of clips per video'
#     )
#     parser.add_argument(
#         '--feat_shape', type=int, nargs='+',default=[1024, 16],
#         help='Feature shape per clip, e.g. "1024 16 7 7"'
#     )
#     parser.add_argument(
#         '--metafile_list', type=str, default="/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_validation_feature_times.pkl",
#         help='Path to pickle file with metadata (must contain video_id, narration_id)'
#     )
#     parser.add_argument(
#         '--out_dir', type=str, default="/mnt/2Tkioxia/yu/feats_ominvoreB_w_tok_npy/val",
#         help='Output directory for per-video .npy files'
#     )
#     args = parser.parse_args()
#     main(args)
