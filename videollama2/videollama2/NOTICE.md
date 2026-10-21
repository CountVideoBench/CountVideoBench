# Third-party code: VideoLLaMA 2

This folder is a copy of the `videollama2/` package of VideoLLaMA 2, the
official implementation of:

> Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen,
> Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, Lidong Bing.
> *VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding
> in Video-LLMs.* arXiv:2406.07476, 2024.

- Upstream repository: https://github.com/DAMO-NLP-SG/VideoLLaMA2
- Upstream commit: `c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414` (2025-01-23)
- License: Apache-2.0, see `LICENSE` in this folder.

## Changes from upstream

Only the files needed for inference are kept: `__init__.py`, `constants.py`,
`mm_utils.py`, `utils.py` and the whole `model/` folder. Upstream's training
code (`train.py`, `videollama2_trainer.py`), benchmark scripts (`eval/`),
demo servers (`serve/`) and `conversation.py` are not included; none of them
is imported at inference time.

The upstream package is not installed as a dependency because its
`pyproject.toml` pins old versions (`torch==2.2.0`, `transformers==4.40.0`,
...) that conflict with the environment used for the paper.

One line is modified:

```diff
--- a/videollama2/model/projector.py
+++ b/videollama2/model/projector.py
@@ -21,7 +21,7 @@
 import torch.nn.functional as F
 from timm.models.regnet import RegStage
 from timm.models.layers import LayerNorm, LayerNorm2d
-from transformers import TRANSFORMERS_CACHE
+#from transformers import TRANSFORMERS_CACHE
```

`TRANSFORMERS_CACHE` no longer exists in transformers 5, so importing the
package fails without this change. The name is only used by
`parse_snapshot_folder`, which loads a separately trained projector and is
never called for the full VideoLLaMA 2 checkpoints. It has no effect on the
computation.
