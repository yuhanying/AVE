import numpy as np
import tqdm
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description=('Create .npy files for the Perception Test dataset'))

###### Things you need to modify ######
parser.add_argument('--feature_file', default="/mnt/2Tkioxia/yu/feats_slowfast/test_vgg/features.npy",type=str, help='Path to extracted features')
parser.add_argument('--pickle_file', default="/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_test_feature_times.pkl",type=str, help='Path to pickle file of feature time intervals')
parser.add_argument('--out_dir', default="/mnt/2Tkioxia/yu/feats_slowfast_npy/test_vgg",type=str, help='Path to save features to')
#######################################


def main(args):
    result_dict = {}
    print("Parsing the features")
    feature = np.load(args.feature_file)
    feature_pandas = pd.read_pickle(args.pickle_file)
    print(len(feature),len(feature_pandas))
    assert len(feature) == len(feature_pandas)

    for i in tqdm.tqdm(range(len(feature_pandas))):
        annotation_id = feature_pandas.iloc[i].name
        # annotation_id = feature_pandas.iloc[i]['narration_id']
        vid_id = feature_pandas.iloc[i]['video_id']
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
        aud_feature = np.stack(args.feature_list, axis=0)
        if len(aud_feature.shape) == 4:
            aud_feature = np.squeeze(aud_feature, axis=1)
        np.save(out_file, aud_feature)

if __name__ == "__main__":
    main(parser.parse_args())
