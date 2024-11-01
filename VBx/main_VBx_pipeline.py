#!/usr/bin/env python

from pyannoteVAD import *
from extract_embedding import *
from clustering import *
import argparse
import os
import sys


parser = argparse.ArgumentParser(description = "VBx Pipeline")
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
parser.add_argument('--input_path', type=str, default="/home3/theanhtran/corpus/DIHARD/DIHARD3/third_dihard_challenge_eval/data/flac/DH_EVAL_0014.flac", help='input path: directory or single file')

# Output file management
parser.add_argument('--output_dir', type=str, default='exp2', help='Output directory')

# VAD parameters
parser.add_argument('--out_vad_lab', type=str, default=None, help='Output VAD file .lab')
parser.add_argument('--pyannote_segementation_token', type=str, default="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", help='Pyannote token')

# extract embedding parameters
parser.add_argument('--out_ark_dir', type=str, default=None, help='Output file .ark')
parser.add_argument('--out_seg_dir', type=str, default=None, help='Output file segment file')
parser.add_argument('--seg_len', type=int, default=144, help='Segment length')
parser.add_argument('--seg_jump', type=int, default=24, help='Segment jump')
parser.add_argument('--weights', type=str, default="models/ResNet101_16kHz/nnet/final.onnx", help='speaker embedding model')
parser.add_argument('--backend', type=str, default="onnx", help='backend model')

# clustering and resegmentation parameters
parser.add_argument('--out_rttm_dir', type=str, default=None, help='Output file rttm')
parser.add_argument('--out_gamma_dir', type=str, default=None, help='Output file gamma npy file')
parser.add_argument('--init', type=str, default="AHC+VB", help='init method')
parser.add_argument('--xvec_transform', type=str, default="models/ResNet101_16kHz/transform.h5", help='Output file .ark')
parser.add_argument('--plda_file', type=str, default="models/ResNet101_16kHz/plda", help='Output file .ark')
parser.add_argument('--threshold', type=float, default=-0.015, help='')
parser.add_argument('--lda_dim', type=int, default=128, help='')
parser.add_argument('--Fa', type=float, default=0.3, help='')
parser.add_argument('--Fb', type=float, default=17, help='')
parser.add_argument('--loopP', type=float, default=0.99, help='')
parser.add_argument('--init_smoothing', type=float, default=0.99, help='')

args = parser.parse_args()

if args.out_vad_lab is None:
    args.out_vad_lab = f"{args.output_dir}/lab"
if args.out_ark_dir is None:
    args.out_ark_dir = f"{args.output_dir}/ark"
if args.out_seg_dir is None:
    args.out_seg_dir = f"{args.output_dir}/segment"
if args.out_rttm_dir is None:
    args.out_rttm_dir = f"{args.output_dir}/rttm"
if args.out_gamma_dir is None:
    args.out_gamma_dir = f"{args.output_dir}/gamma_npy"

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

def create_wav_scp(input_path, output_file="wav.scp"):
    # Check if the input path is a file or directory
    audio_files = []
    if os.path.isdir(input_path):
        # Collect all .wav and .flac files in the directory and subdirectories
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".wav") or file.endswith(".flac"):
                    audio_files.append(os.path.join(root, file))
    elif os.path.isfile(input_path) and (input_path.endswith(".wav") or input_path.endswith(".flac")):
        # Single audio file case
        audio_files.append(input_path)
    else:
        print("Invalid input path. Provide a .wav or .flac file or directory containing such files.")
        sys.exit(1)

    # Write to wav.scp file in the specified format
    with open(output_file, 'w') as f:
        for audio_file in audio_files:
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            f.write(f"{filename} {audio_file}\n")
    print(f"wav.scp file created with {len(audio_files)} entries.")

wav_scp_file = f'{args.output_dir}/wav.scp'

create_wav_scp(args.input_path, wav_scp_file)

pyannote_vad(wav_scp_file, args)

extract_embedding(wav_scp_file, args)

clustering_vbhmm_resegmentation(wav_scp_file, args)