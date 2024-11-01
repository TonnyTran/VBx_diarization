#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import time
import onnxruntime
import soundfile as sf
import torch.backends
import features
from models.resnet import *
import argparse
import os
import kaldi_io
import numpy as np
torch.backends.cudnn.enabled = False
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
import os
from overlap_utils import *
from datetime import datetime

def initialize_gpus(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            logger.info(f'Start: {self.name}: ')

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info(f'End:   {self.name}: Elapsed: {time.time() - self.tstart} seconds')
        else:
            logger.info(f'End:   {self.name}: ')


def get_embedding(fea, model, label_name=None, input_name=None, backend='pytorch'):
    if backend == 'pytorch':
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)
        spk_embeds = model(data)
        return spk_embeds.data.cpu().numpy()[0]
    elif backend == 'onnx':
        return model.run([label_name],
                         {input_name: fea.astype(np.float32).transpose()
                         [np.newaxis, :, :]})[0].squeeze()


def extract_embedding(wav_scp_file, args):
    if not os.path.exists(args.weights):
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')
    device = ''
    if args.gpus != '':
        logger.info(f'Using GPU: {args.gpus}')

        # gpu configuration
        initialize_gpus(args)
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    model, label_name, input_name = '', None, None

    if args.backend == 'onnx':
        model = onnxruntime.InferenceSession(args.weights)
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

    else:
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')

    if not os.path.exists(args.out_seg_dir):
        os.makedirs(args.out_seg_dir)
    if not os.path.exists(args.out_ark_dir):
        os.makedirs(args.out_ark_dir)

    with torch.no_grad():
        # process each file
        with open(wav_scp_file, 'r') as wav_scp:
            for line in wav_scp:
                filename, wav_path = line.strip().split(maxsplit=1)
                with open(f'{args.out_seg_dir}/{filename}.seg', 'w') as seg_file:
                    with open(f'{args.out_ark_dir}/{filename}.ark', 'wb') as ark_file:
                        with Timer(f'Processing file {filename}'):
                            signal, samplerate = sf.read(wav_path)
                            labs = np.atleast_2d((np.loadtxt(f'{args.out_vad_lab}/{filename}.lab',usecols=(0, 1)) * samplerate).astype(int))
                            if samplerate == 8000:
                                noverlap = 120
                                winlen = 200
                                window = features.povey_window(winlen)
                                fbank_mx = features.mel_fbank_mx(winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=3700, htk_bug=False)
                            elif samplerate == 16000:
                                noverlap = 240
                                winlen = 400
                                window = features.povey_window(winlen)
                                fbank_mx = features.mel_fbank_mx(winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
                            else:
                                raise ValueError(f'Only 8kHz and 16kHz are supported. Got {samplerate} instead.')

                            LC = 150
                            RC = 149
                            np.random.seed(3)  # for reproducibility
                            signal = features.add_dither((signal*2**15).astype(int))
                            for segnum in range(len(labs)):
                                seg = signal[labs[segnum, 0]:labs[segnum, 1]]
                                if seg.shape[0] > 0.01*samplerate:  # process segment only if longer than 0.01s
                                        # Mirror noverlap//2 initial and final samples
                                    seg = np.r_[seg[noverlap // 2 - 1::-1],
                                                seg, seg[-1:-winlen // 2 - 1:-1]]
                                    fea = features.fbank_htk(seg, window, noverlap, fbank_mx,
                                                                USEPOWER=True, ZMEANSOURCE=True)
                                    fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)

                                    slen = len(fea)
                                    start = -args.seg_jump

                                    for start in range(0, slen - args.seg_len, args.seg_jump):
                                        data = fea[start:start + args.seg_len]                                
                                        xvector = get_embedding(data, model, label_name=label_name, input_name=input_name, backend=args.backend)
                                        key = f'{filename}_{segnum:04}-{start:08}-{(start + args.seg_len):08}'
                                        if np.isnan(xvector).any():
                                            logger.warning(f'NaN found, not processing: {key}{os.linesep}')
                                        else:
                                            seg_start = round(labs[segnum, 0] / float(samplerate) + start / 100.0, 3)
                                            seg_end = round(
                                                labs[segnum, 0] / float(samplerate) + start / 100.0 + args.seg_len / 100.0, 3
                                            )
                                            seg_file.write(f'{key} {filename} {seg_start} {seg_end}{os.linesep}')
                                            kaldi_io.write_vec_flt(ark_file, xvector, key=key)

                                    if slen - start - args.seg_jump >= 10:
                                        data = fea[start + args.seg_jump:slen]                               
                                        xvector = get_embedding(
                                                data, model, label_name=label_name, input_name=input_name, backend=args.backend)

                                        key = f'{filename}_{segnum:04}-{(start + args.seg_jump):08}-{slen:08}'

                                        if np.isnan(xvector).any():
                                            logger.warning(f'NaN found, not processing: {key}{os.linesep}')
                                        else:
                                            seg_start = round(
                                                labs[segnum, 0] / float(samplerate) + (start + args.seg_jump) / 100.0, 3
                                            )
                                            seg_end = round(labs[segnum, 1] / float(samplerate), 3)
                                            seg_file.write(f'{key} {filename} {seg_start} {seg_end}{os.linesep}')
                                            kaldi_io.write_vec_flt(ark_file, xvector, key=key)
    print(datetime.now().time(), ": Done extracting embeddings")


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

#     args = parser.parse_args()

#     if args.out_vad_lab is None:
#         args.out_vad_lab = f"{args.output_dir}/lab"
#     if args.out_ark_dir is None:
#         args.out_ark_dir = f"{args.output_dir}/ark"
#     if args.out_seg_dir is None:
#         args.out_seg_dir = f"{args.output_dir}/segment"

#     wav_scp_file = f'{args.output_dir}/wav.scp'

#     extract_embedding(wav_scp_file, args)
                  