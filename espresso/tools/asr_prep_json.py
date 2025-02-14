#!/usr/bin/env python3
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import logging
import sys


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("espresso.tools.asr_prep_json")


def read_file(ordered_dict, key, dtype, *paths):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, val = line.strip().split(None, 1)
                if utt_id in ordered_dict:
                    assert key not in ordered_dict[utt_id], \
                        "Duplicate utterance id " + utt_id + " in " + key
                    ordered_dict[utt_id].update({key: dtype(val)})
                else:
                    ordered_dict[utt_id] = {key: val}
    return ordered_dict


def main():
    parser = argparse.ArgumentParser(
        description="Wrap all related files of a dataset into a single json file"
    )
    # fmt: off
    audio_group = parser.add_mutually_exclusive_group(required=True)
    audio_group.add_argument("--feat-files", nargs="+", default=None,
                             help="path(s) to feats.scp feature file(s) from Kaldi")
    audio_group.add_argument("--wave-files", nargs="+", default=None,
                             help="path(s) to raw waveform file(s), where each entry has the format '<utt-id> <file-path>'")
    audio_group.add_argument("--command-files", nargs="+", default=None,
                             help="path(s) to wav.scp file(s) from Kaldi")
    parser.add_argument("--token-text-files", nargs="+", default=None,
                        help="path(s) to token_text file(s)")
    parser.add_argument("--text-files", nargs="+", default=None,
                        help="path(s) to text file(s)")
    parser.add_argument("--numerator-fst-files", nargs="+", default=None,
                        help="path(s) to numerator fst file(s)")
    parser.add_argument("--alignment-files", nargs="+", default=None,
                        help="path(s) to alignment file(s)")
    parser.add_argument("--utt2num-frames-files", nargs="+", default=None,
                        help="path(s) to utt2num_frames file(s)")
    parser.add_argument("--output", required=True, type=argparse.FileType("w"),
                        help="path to save json output")
    args = parser.parse_args()
    # fmt: on

    obj = OrderedDict()
    if args.feat_files is not None:
        obj = read_file(obj, "feat", str, *(args.feat_files))
    elif args.wave_files is not None:
        obj = read_file(obj, "wave", str, *(args.wave_files))
    else:
        assert args.command_files is not None
        obj = read_file(obj, "command", str, *(args.command_files))
    if args.token_text_files is not None:
        obj = read_file(obj, "token_text", str, *(args.token_text_files))
    if args.text_files is not None:
        obj = read_file(obj, "text", str, *(args.text_files))
    if args.numerator_fst_files is not None:
        obj = read_file(obj, "numerator_fst", str, *(args.numerator_fst_files))
    if args.alignment_files is not None:
        obj = read_file(obj, "alignment", str, *(args.alignment_files))
    if args.utt2num_frames_files is not None:
        obj = read_file(obj, "utt2num_frames", int, *(args.utt2num_frames_files))

    json.dump(obj, args.output, indent=4)
    logger.info("Dumped {} examples in {}".format(len(obj), args.output.name))


if __name__ == "__main__":
    main()
