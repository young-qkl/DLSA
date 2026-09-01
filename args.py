import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='DLSA training and evaluation')
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('--text_decoder_stages', default='tg2_tg1',
                        choices=['none', 'tg1', 'tg2_tg1', 'tg4_tg3', 'all'],
                        help='which decoder stages use text-guided decoder blocks')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--bert_tokenizer', default='bert-base-uncased', help='BERT tokenizer')
    parser.add_argument('--ck_bert', default='./bert-base-uncased', help='pre-trained BERT weights')
    parser.add_argument('--dataset', default='rrsisd', choices=['rrsisd', 'refsegrs'])
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='Only needs specified when testing,'
                             'whether the weights to be loaded are from a DDP-trained model')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--fusion_drop', default=0.0, type=float, help='dropout rate for PWAMs')
    parser.add_argument('--img_size', default=480, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int,default=0,help='local rank for DistributedDataParallel')
    parser.add_argument('--lr', default=0.00003, type=float, help='the initial learning rate')
    parser.add_argument('--mha', default='', help='If specified, should be in the format of a-b-c-d, e.g., 4-4-4-4,'
                                                  'where a, b, c, and d refer to the numbers of heads in stage-1,'
                                                  'stage-2, stage-3, and stage-4 PWAMs')
    parser.add_argument('--model', default='lavt_one', help='model: lavt, l avt_one')
    parser.add_argument('--model_id', default='RMSIN', help='name to identify the model')
    parser.add_argument('--output-dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--pretrained_swin_weights', default='./pretrained_weights/swin_base_patch4_window12_384_22k.pth',
                        help='path to pre-trained Swin backbone weights')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data', help='REFER dataset root directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--refsegrs_data_root', default='./datasets/RefSegRS',
                        help='RefSegRS dataset root directory')
    parser.add_argument('--split', default='test', help='only used when testing')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the datasset is G-Ref (RefCOCOg)')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, or large variants of the Swin Transformer')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--window12', action='store_true',
                        help='only needs specified when testing,'
                             'when training, window size is inferred from pre-trained weights file name'
                             '(containing \'window12\'). Initialize Swin with window size 12 instead of the default 7.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--vis_save_dir', default=None, help='folder to save visualizations')
    parser.add_argument('--sg_mode', default='language', choices=['language', 'avg', 'fixed3', 'fixed5', 'fixed7'],
                        help='multi-scale conv fusion strategy: use language or fixed kernel size')
    parser.add_argument('--sgamc_layout', default='original', choices=['original', 'legacy'],
                        help='checkpoint-compatible SgAMC parameter layout')
    parser.add_argument('--log-dir', default='./logs', help='Directory to save training logs')
    # args
    parser.add_argument('--sgamc_tau', type=float, default=0.7)
    parser.add_argument('--sgamc_learnable_tau', action='store_true', default=True)
    parser.add_argument('--sgamc_entropy_lambda', type=float, default=5e-3)  # 建议 1e-3 ~ 1e-2 网格
    parser.add_argument('--tau_test', type=float, default=None, help='override temperature at inference')
    parser.add_argument('--gamma_test', type=float, default=None, help='override suppressive gamma at inference')
    parser.add_argument('--topk_test', type=int, default=None, help='use top-k branch mixing at inference')
    parser.add_argument('--mono_channel_gate', action='store_true', help='average gate across channels at inference')
    parser.add_argument('--init_weights', default='', type=str,
                        help='warm-start checkpoint for LA-SgAMC finetuning')

    parser.add_argument('--ft_stage', default='A', choices=['A', 'B', 'full'],
                        help='A: only new LA params + alpha_fc; B: all sgAMC + alpha_fc; full: keep original training scope')

    # Text-Guided Decoder v2 / ablation controls
    parser.add_argument('--use_text_decoder', action='store_true',
                        help='enable identity-safe text-guided decoder blocks')

    parser.add_argument('--text_decoder_variant', default='glf_adapt',
                        choices=['attn', 'glf', 'glf_adapt'],
                        help='text decoder variant: attn=TD-v2, glf=TD-v3 GLF-lite, glf_adapt=ALTI/TD-v4A')

    parser.add_argument('--alti_ablation', default='none',
                        choices=['none', 'no_alpha', 'no_local', 'no_scale'],
                        help='component ablation for ALTI: '
                             'none=full ALTI, no_alpha=remove sample-wise gate, '
                             'no_local=remove local gate, no_scale=remove residual scale')

    parser.add_argument('--td_res_scale_init', type=float, default=0.0,
                        help='initial residual scale for identity-safe text decoder')

    parser.add_argument('--td_gate_bias_init', type=float, default=-4.0,
                        help='initial gate bias for identity-safe text decoder')

    parser.add_argument('--td_alpha_bias_init', type=float, default=-2.0,
                        help='initial bias for adaptive alpha gate in ALTI/TD-v4A')

    parser.add_argument('--td_alpha_weight_std', type=float, default=0.0,
                        help='std for initializing adaptive alpha gate final layer weight in ALTI/TD-v4A')

    parser.add_argument('--td_uncertainty_gate', action='store_true',
                        help='enable uncertainty-aware visual statistics in TD-v4A adaptive alpha gate')


    parser.add_argument('--use_decoder_sgamc', action='store_true',
                        help='optional: enable legacy decoder-side SgAMC; default keeps original raw path unchanged')

    parser.add_argument('--sgamc_kernel_ablation', default='none',
                        choices=['none', 'wo_k1', 'wo_k3', 'wo_k5', 'wo_k7'],
                        help='SgAMC kernel branch ablation. none keeps {1,3,5,7}; '
                             'wo_k1/wo_k3/wo_k5/wo_k7 remove the corresponding kernel branch.')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
