# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Union

import torch
from fairseq.data import Dictionary, encoders
from fairseq.file_io import PathManager
from omegaconf import DictConfig

# will automatically load modules defined from there
from espresso.data import encoders as encoders_espresso


class AsrDictionary(Dictionary):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        space="<space>",
        enable_bos=False,
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word, self.space_word = (
            bos, unk, pad, eos, space
        )
        self.symbols = []
        self.count = []
        self.indices = {}
        # no bos added to the dictionary by default
        if enable_bos:
            self.bos_index = self.add_symbol(self.bos_word)
        self.pad_index = self.add_symbol(pad, n=0)
        self.eos_index = self.add_symbol(eos, n=0)
        self.unk_index = self.add_symbol(unk, n=0)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s, n=0)
        self.nspecial = len(self.symbols)
        self.non_lang_syms = None
        self.tokenizer = None
        self.bpe = None

    def bos(self):
        """Disallow beginning-of-sentence symbol if not exists"""
        if hasattr(self, "bos_index"):
            return self.bos_index
        raise NotImplementedError

    def space(self):
        """Helper to get index of space symbol"""
        return self.space_index

    @classmethod
    def load(cls, f, enable_bos=False, f_non_lang_syms=None):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```

        Optionally add bos symbol to the dictionary

        Identifies the space symbol if it exists, by obtaining its index
        (space_index=-1 if no space symbol)

        Loads non_lang_syms from another text file, if it exists, with one
        symbol per line
        """
        d = cls(enable_bos=enable_bos)
        d.add_from_file(f)

        d.space_index = d.indices.get(d.space_word, -1)

        if f_non_lang_syms is not None:
            assert isinstance(f_non_lang_syms, str)
            try:
                with open(PathManager.get_local_path(f_non_lang_syms), "r", encoding="utf-8") as fd:
                    non_lang_syms = [x.rstrip() for x in fd.readlines()]
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(fd)
                )

            for sym in non_lang_syms:
                assert (
                    d.index(sym) != d.unk()
                ), "{} in {} is not in the dictionary".format(sym, f_non_lang_syms)
            d.non_lang_syms = non_lang_syms

        return d

    def dummy_sentence(self, length):
        # sample excluding special symbols
        t = torch.Tensor(length).uniform_(self.nspecial, len(self)).long()
        t[-1] = self.eos()
        return t

    def build_tokenizer(self, cfg: Union[DictConfig, Namespace]):
        self.tokenizer = encoders.build_tokenizer(cfg)

    def build_bpe(self, cfg: Union[DictConfig, Namespace]):
        if (
            (isinstance(cfg, DictConfig) and cfg._name == "characters_asr")
            or (isinstance(cfg, Namespace) and getattr(cfg, "bpe", None) == "characters_asr")
        ):
            self.bpe = encoders.build_bpe(
                cfg, space_symbol=self.space_word, non_lang_syms=self.non_lang_syms
            )
        else:
            self.bpe = encoders.build_bpe(cfg)

    def wordpiece_encode(self, x):
        if self.tokenizer is not None:
            x = self.tokenizer.encode(x)
        if self.bpe is not None:
            x = self.bpe.encode(x)
        return x

    def wordpiece_decode(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x
