#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
import os
import argparse
from datetime import datetime

def pyannote_vad(wav_scp_file, args):
    # load model
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=args.pyannote_segementation_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    HYPER_PARAMETERS = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)

    if not os.path.exists(args.out_vad_lab):
        os.makedirs(args.out_vad_lab)

    # process VAD for each file
    with open(wav_scp_file, 'r') as wav_scp:
        for line in wav_scp:
            filename, wav_path = line.strip().split(maxsplit=1)
            vad = pipeline(wav_path)

            with open(f'{args.out_vad_lab}/{filename}.lab','w') as vad_lab:
                vad_lab.write(vad.to_lab())
            vad_lab.close()
            assert os.path.exists(args.out_vad_lab), f"Lab File processing didnt complete: {args.out_vad_lab}"
    
    print(datetime.now().time(), ": Done VAD processing")

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

#     args = parser.parse_args()

#     if args.out_vad_lab is None:
#         args.out_vad_lab = f"{args.output_dir}/lab"
    
#     wav_scp_file = f'{args.output_dir}/wav.scp'

#     pyannote_vad(wav_scp_file, args)