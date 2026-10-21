# Third-party code: AdaReTaKe

This folder is a copy of the `retake/` package of AdaReTaKe, the official
implementation of:

> Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, Liqiang Nie.
> *AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for
> Video-language Understanding.* arXiv:2503.12559, 2025.

- Upstream repository: https://github.com/iLearn-Lab/ACL25-AdaReTaKe
  (formerly `SCZwangxiao/video-FlexReduc`)
- Upstream commit: `a16a662606ac4a979cbaf4eec366beaeb94f4fb7` (2025-04-23)
- License: MIT, see `LICENSE` in this folder. Note that the upstream README
  also states "Research use only — Commercial applications require explicit
  permission."

## Changes from upstream

Only the files needed to run Qwen2.5-VL are kept: `monkeypatch.py`,
`qwen2_vl.py`, `qwen2_5_vl.py`, `llava_onevision.py`, `longvideo_cache.py` and
`visual_compression.py`. `monkeypatch.py` imports the three model modules, so
all of them are required even though only Qwen2.5-VL is used. Upstream's
`dataset_utils.py` and `infer_eval.py` (benchmark loaders for VideoMME, MLVU,
etc.) are not included.

One file is modified:

```diff
--- a/retake/llava_onevision.py
+++ b/retake/llava_onevision.py
@@ -1,3 +1,6 @@
+from __future__ import annotations
+
+
 import math
```

(plus two blank lines further down, with no other change). This defers the evaluation of type annotations. Without it, importing
`retake.monkeypatch` fails with `NameError: name 'FlashAttentionKwargs' is not
defined` whenever flash-attn cannot be loaded, because that name is only
imported when flash-attn is available. It has no effect on the computation.
