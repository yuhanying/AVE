import argparse
import random

from time_interval_machine.utils.misc import str2bool
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description=('Train Audio-Visual Transformer on Sequence ' +
                                              'of actions from untrimmed video'))


    # ------------------------------ Dataset -------------------------------
    parser.add_argument('--video_data_path', type=Path, default=Path('/mnt/2Tkioxia/yu/feats_ominvoreB_w_tok_npy'))
    parser.add_argument('--audio_data_path', type=Path, default=Path('/mnt/2Tkioxia/yu/feats_slowfast_npy'))
    parser.add_argument('--video_train_action_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_train.pkl'))
    parser.add_argument('--video_val_action_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_validation.pkl'))
    parser.add_argument('--video_test_action_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_test.pkl'))
    parser.add_argument('--video_train_context_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_train_feature_times.pkl'))
    parser.add_argument('--video_val_context_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_validation_feature_times.pkl'))
    parser.add_argument('--video_test_context_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_test_feature_times.pkl'))
    parser.add_argument('--audio_train_action_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_train.pkl'))
    parser.add_argument('--audio_val_action_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_validation.pkl'))
    parser.add_argument('--audio_test_action_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_test.pkl'))
    parser.add_argument('--audio_train_context_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_train_feature_times.pkl'))
    parser.add_argument('--audio_val_context_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_validation_feature_times.pkl'))
    parser.add_argument('--audio_test_context_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_1_second_test_feature_times.pkl'))
    parser.add_argument('--video_info_pickle', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/annotations/annotations/AVE/AVE_video_info.pkl'))
    parser.add_argument('--ckpt_path', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/output/models'))
    parser.add_argument('--ckpt_path_best', type=Path, default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/output/models/best_model.pth'))
    
    parser.add_argument('--include_verb_noun', type=str2bool, default=False)
    parser.add_argument('--dataset',
                        default='ave',
                        choices=['epic', 'perception', 'ave'],
                        help='Dataset to train/validate on'
                    )
    parser.add_argument('--num_feats', type=int, default=25)
    parser.add_argument('--feat_stride',
                        default=2,
                        type=int,
                        help='Context hop'
                    )
    parser.add_argument('--feat_gap',
                        default=0.2,
                        type=float,
                        help='Time between extracted features'
                    )
    parser.add_argument('--window_stride',
                        default=1.0,
                        type=float,
                        help='Stride of input window in seconds'
                    )
    parser.add_argument('--data_modality',
                        type=str,
                        default='audio_visual',
                        choices=['visual', 'audio', 'audio_visual'],
                        help='Modality to train/validate model on'
                    )
    # ------------------------------ Model ---------------------------------
    parser.add_argument('--num_class', default=([97, 300, 3806], 44))
    parser.add_argument('--visual_input_dim', type=int, default=1024)
    parser.add_argument('--audio_input_dim', type=int, default=2304)
    parser.add_argument('--d_model', type=int, default=384)
    parser.add_argument('--feedforward_scale', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--enc_dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.1)
    parser.add_argument('--seq_dropout', type=float, default=0.1)
    parser.add_argument('--model_modality',
                        default='audio_visual',
                        type=str,
                        choices=['visual', 'audio', 'audio_visual'],
                        help='Modality of the input features'
                    )
    parser.add_argument('--apply_feature_pooling', 
                        default=False,
                        type=str2bool, 
                        help='Apply Audio-Guided Spatial Pooling to input features'
                    )
    # ------------------------------ Train ----------------------------------
    parser.add_argument('--finetune_epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of video epochs to run'
                    )
    parser.add_argument('--warmup_epochs',
                        default=2,
                        type=int,
                        metavar='N',
                        help='number of epochs to run warmup'
                    )
    parser.add_argument('-b', '--batch-size',
                        default=8,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)'
                    )
    parser.add_argument('--pretrained_model',
                        default='/mnt/2Tkioxia/yu/feats_psp_by_video/ave.pth (1).tar',
                        type=str,
                        help='pretrained model weights'
                    )
    parser.add_argument('--lambda_drloc',
                        default=-1,
                        type=float,
                        help='lambda for drloc'
                    )
    parser.add_argument('--mixup_alpha',
                        default=0.5,
                        type=float,
                        help='alpha for mixup'
                    )
    parser.add_argument('--lambda_audio',
                        default=1,
                        type=float,
                        help='alpha for audio'
                    )
    parser.add_argument('--m_drloc',
                        default=32,
                        type=int,
                        help='m for drloc'
                    )
    parser.add_argument('--enable_amp', type=str2bool, default=True)
    parser.add_argument('--early_stop_period', type=int, default=5)
    # ------------------------------ Optimizer ------------------------------
    parser.add_argument('--lr', '--learning-rate',
                        default=5e-5,
                        type=float,
                        metavar='LR',
                        help='initial learning rate'
                    )
    parser.add_argument('--weight_decay', '--wd',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 5e-4)'
                    )
    # ---------------------------- Run Flags ---------------------------------
    parser.add_argument('--train', action='store_true',default=1)
    parser.add_argument('--validate', action='store_true',default=0)
    parser.add_argument('--extract_feats', action='store_true')
    # ------------------------------ Misc ------------------------------------
    parser.add_argument('--output_dir', type=Path,default=Path('/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/output'))
    parser.add_argument('--enable_wandb_log', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help='Random Seed')
    parser.add_argument('--print-freq', '-p',
                        default=100,
                        type=int,
                        metavar='N',
                        help='Frequency to log iteration information'
                    )
    # ---------------------------- Resources ---------------------------------
    parser.add_argument('-j', '--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 8)'
                    )
    parser.add_argument('--pin-memory',
                        default=True,
                        type=str2bool,
                        help='Pin memory in dataloader'
                    )
    # --------------------------- Distributed --------------------------------
    parser.add_argument("--shard_id",
                        default=0,
                        type=int,
                        help="The shard id of current node, Starts from 0 to num_shards - 1"
                    )
    parser.add_argument("--num_shards",
                        default=1,
                        type=int,
                        help="Number of shards using by the job"
                    )
    parser.add_argument("--init_method",
                        default="tcp://localhost:9999",
                        type=str,
                        help="Initialization method, includes TCP or shared file-system"
                    )
    parser.add_argument('--num-gpus',
                        default=1,
                        type=int,
                        help='number of GPUs to train model on'
                    )
    parser.add_argument("--dist_backend",
                        default="nccl",
                        type=str,
                        help="Distributed backend to use"
                    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    if args.validate:
        assert args.pretrained_model != ""
        
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)

    if not args.include_verb_noun and isinstance(args.num_class[0], list):
        args.num_class = (args.num_class[0][2], args.num_class[1])

    if args.dataset == 'perception':
        args.num_class = (63, 17)
    
    if args.dataset == 'ave':
        args.num_class = (29, 29)

    return args
