# AVE

## how to run

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
The features used for this project can be extracted by following the instructions in the `feature_extractors` folder.  

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
因為檔案有35G，我先放在NAS上(`/MB605/feats_npy_ave`）

