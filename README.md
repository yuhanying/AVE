# AVE

## how to run
To train TIM with Omnivore + AuditorySlowFast features, run
```[bash]
python recognition/scripts/run_net.py \
--train \
--output_dir /path/to/output \
--video_data_path /path/to/feats_ominvoreB_w_tok_npy \
--video_train_action_pickle /path/to/annotations/AVE/AVE_train.pkl \
--video_val_action_pickle /path/to/annotations/AVE/AVE_validation.pkl \
--video_test_action_pickle /path/to/annotations/AVE/AVE_test.pkl \
--video_train_context_pickle /path/to/annotations/AVE/AVE_1_second_train_feature_times.pkl \
--video_val_context_pickle /path/to/annotations/AVE/AVE_1_second_validation_feature_times.pkl \
--video_test_context_pickle /path/to/annotations/AVE/AVE_1_second_test_feature_times.pkl \
--visual_input_dim 1024 \
--audio_data_path /path/to/feats_slowfast_npy \
--audio_train_action_pickle /path/to/annotations/AVE/AVE_train.pkl \
--audio_val_action_pickle /path/to/annotations/AVE/AVE_validation.pkl \
--audio_test_action_pickle /path/to/annotations/AVE/AVE_test.pkl \
--audio_train_context_pickle /path/to/annotations/AVE/AVE_1_second_train_feature_times.pkl \
--audio_val_context_pickle /path/to/annotations/AVE/AVE_1_second_validation_feature_times.pkl \
--audio_test_context_pickle /path/to/annotations/AVE/AVE_1_second_test_feature_times.pkl \
--audio_input_dim 2304 \
--video_info_pickle /path/to/annotations/AVE/AVE_video_info.pkl \
--dataset ave \
--feat_stride 2 \
--num_feats 25 \
--feat_dropout 0.1 \
--d_model 384 \
--batch-size 8 \
--lr 5e-5 \
--lambda_drloc -1 \
--mixup_alpha 0.5 \
--lambda_audio 1 \
--early_stop_period 4 \
--include_verb_noun False
```

## best model
[here](https://drive.google.com/drive/folders/1ze6FTZu1OS6SbSW0xy8UBus9NQ0bxPXs?usp=sharing)

## ground-truth
We provide the necessary ground-truth files for all datasets [here](https://drive.google.com/drive/folders/1rPTiH5uPqxQ_wgvUixmBHHYl2J_ATSgL?usp=sharing).

The link contains a zip containing ground truth data for each dataset, consisting of:

- The training split ground truth
- The validation split ground truth
- The video metadata of the dataset
The feature time intervals for training and valdiation splits

## Post processing AVE features
The features used for this project can be extracted by following the instructions in the `feats_extract` folder.  

You need to post-process the features with numpy files. The directory structure should be something like below:

```[bash]
output_dir
├── train     
    ├──004KfU7bgyg.npy
    ├──0095-_8T5ZY.npy
    ...    
├── val
    ├──00N83yxKYUI.npy
    ├──01eOqSIF9PE.npy
    ...       
├── test
    ├──024vJIboFg4.npy
    ├──02rnKVawh0E.npy
    ...                
```

To make this, run the following code. You need to change the `feature_list`, `metafile_list` and `out_dir` on your own.

```[python]
python utils/ave/make_npyfiles.py
```

the feature extracted(.npy files) from omnivore (visual) and slowfast (audio) can be download here.  
這邊有先提取好了，因為檔案有35G，我先放在NAS上(`/MB605/feats_npy_ave`）

