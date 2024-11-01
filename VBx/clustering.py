#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch.backends
from models.resnet import *
import os
import itertools
import fastcluster
import h5py
import kaldi_io
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh
from diarization_lib import read_xvector_timing_dict, l2_norm,cos_similarity, twoGMMcalib_lin, merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VBx import VBx
torch.backends.cudnn.enabled = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from datetime import datetime

def write_output(fp,file_name, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


def clustering_vbhmm_resegmentation(wav_scp_file, args):
    # if len(wav_path)>=0 and os.path.exists(wav_path):
    #     full_name = os.path.basename(wav_path)
    #     filename = os.path.splitext(full_name)[0]
    #     print(filename)
    # else:
    #     raise ValueError('Wrong path parameters provided (or not provided at all)')
    
    assert 0 <= args.loopP <= 1, f'Expecting config loopP between 0 and 1, got {args.loopP} instead.'
    kaldi_plda = read_plda(args.plda_file)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    # process VAD for each file
    with open(wav_scp_file, 'r') as wav_scp:
        for line in wav_scp:
            filename, wav_path = line.strip().split(maxsplit=1)

            segs_dict = read_xvector_timing_dict(f'{args.out_seg_dir}/{filename}.seg')
            # Open ark file with x-vectors and in each iteration of the following
            # for-loop read a batch of x-vectors corresponding to one recording
            arkit = kaldi_io.read_vec_flt_ark(f'{args.out_ark_dir}/{filename}.ark')
            # group xvectors in ark by recording name
            recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])
            for file_name, segs in recit:
                print(file_name)
                seg_names, xvecs = zip(*segs)
                x = np.array(xvecs)

                with h5py.File(args.xvec_transform, 'r') as f:
                    mean1 = np.array(f['mean1'])
                    mean2 = np.array(f['mean2'])
                    lda = np.array(f['lda'])
                    x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

                if args.init == 'AHC' or args.init.endswith('VB'):
                    if args.init.startswith('AHC'):
                        # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                        # similarities between all x-vectors)
                        scr_mx = cos_similarity(x)
                        # Figure out utterance specific args.threshold for AHC
                        thr, _ = twoGMMcalib_lin(scr_mx.ravel())
                        # output "labels" is an integer vector of speaker (cluster) ids
                        scr_mx = squareform(-scr_mx, checks=False)
                        lin_mat = fastcluster.linkage(
                            scr_mx, method='average', preserve_input='False')
                        del scr_mx
                        adjust = abs(lin_mat[:, 2].min())
                        lin_mat[:, 2] += adjust
                        labels1st = fcluster(lin_mat, -(thr + args.threshold) + adjust,
                            criterion='distance') - 1
                    if args.init.endswith('VB'):
                        # Smooth the hard labels obtained from AHC to soft assignments
                        # of x-vectors to speakers
                        qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
                        qinit[range(len(labels1st)), labels1st] = 1.0
                        qinit = softmax(qinit * args.init_smoothing, axis=1)
                        fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]
                        q, sp, L = VBx(
                            fea, plda_psi[:args.lda_dim],
                            pi=qinit.shape[1], gamma=qinit,
                            maxIters=40, epsilon=1e-6,
                            loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb)
                        mkdir_p(args.out_gamma_dir)
                        np.save(f'{args.out_gamma_dir}/{file_name}.npy', q) #  timeframe * Speaker posterior probabilities ../
                        labels1st = np.argsort(-q, axis=1)[:, 0]
                        if q.shape[1] > 1:
                            labels2nd = np.argsort(-q, axis=1)[:, 1]
                else:
                    raise ValueError('Wrong option for args.initialization.')

                assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
                start, end = segs_dict[file_name][1].T

                starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st)
                mkdir_p(args.out_rttm_dir)
                with open(os.path.join(args.out_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                    write_output(fp, file_name, out_labels, starts, ends)
    print(datetime.now().time(), ": Done Clustering!!!")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description = "VBx Pipeline")
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
#     parser.add_argument('--input_path', type=str, default="/home3/theanhtran/corpus/DIHARD/DIHARD3/third_dihard_challenge_eval/data/flac/DH_EVAL_0014.flac", help='input path: directory or single file')

#     # Output file management
#     parser.add_argument('--output_dir', type=str, default='exp2', help='Output directory')

#     # VAD parameters
#     parser.add_argument('--out_vad_lab', type=str, default=None, help='Output VAD file .lab')
#     parser.add_argument('--pyannote_segementation_token', type=str, default="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", help='Pyannote token')

#     # extract embedding parameters
#     parser.add_argument('--out_ark_dir', type=str, default=None, help='Output file .ark')
#     parser.add_argument('--out_seg_dir', type=str, default=None, help='Output file segment file')
#     parser.add_argument('--seg_len', type=int, default=144, help='Segment length')
#     parser.add_argument('--seg_jump', type=int, default=24, help='Segment jump')
#     parser.add_argument('--weights', type=str, default="models/ResNet101_16kHz/nnet/final.onnx", help='speaker embedding model')
#     parser.add_argument('--backend', type=str, default="onnx", help='backend model')

#     # clustering and resegmentation parameters
#     parser.add_argument('--out_rttm_dir', type=str, default=None, help='Output file rttm')
#     parser.add_argument('--out_gamma_dir', type=str, default=None, help='Output file gamma npy file')
#     parser.add_argument('--init', type=str, default="AHC+VB", help='init method')
#     parser.add_argument('--xvec_transform', type=str, default="models/ResNet101_16kHz/transform.h5", help='Output file .ark')
#     parser.add_argument('--plda_file', type=str, default="models/ResNet101_16kHz/plda", help='Output file .ark')
#     parser.add_argument('--threshold', type=float, default=-0.015, help='')
#     parser.add_argument('--lda_dim', type=int, default=128, help='')
#     parser.add_argument('--Fa', type=float, default=0.3, help='')
#     parser.add_argument('--Fb', type=float, default=17, help='')
#     parser.add_argument('--loopP', type=float, default=0.99, help='')
#     parser.add_argument('--init_smoothing', type=float, default=0.99, help='')

#     args = parser.parse_args()

#     if args.out_vad_lab is None:
#         args.out_vad_lab = f"{args.output_dir}/lab"
#     if args.out_ark_dir is None:
#         args.out_ark_dir = f"{args.output_dir}/ark"
#     if args.out_seg_dir is None:
#         args.out_seg_dir = f"{args.output_dir}/segment"
#     if args.out_rttm_dir is None:
#         args.out_rttm_dir = f"{args.output_dir}/rttm"
#     if args.out_gamma_dir is None:
#         args.out_gamma_dir = f"{args.output_dir}/gamma_npy"

#     wav_scp_file = f'{args.output_dir}/wav.scp'

#     clustering_vbhmm_resegmentation(wav_scp_file, args)