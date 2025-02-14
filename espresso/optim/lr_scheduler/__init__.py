# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if not file.startswith("_") and not file.startswith(".") and file.endswith(".py"):
        file_name = file[: file.find(".py")]
        importlib.import_module("espresso.optim.lr_scheduler." + file_name)
