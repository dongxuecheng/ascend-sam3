# OmniCrop provenance

- Author / upstream: Leon, https://github.com/leon0514/OmniCrop
- Pinned commit: `e9faf56f45df2ee69d028dfaa91b17b6b806b8dd`
- Vendored file: upstream `src/OmniCrop.hpp`, algorithm unmodified.
- Upstream package metadata (`pyproject.toml`, version 1.0.3) declares
  `license = {text = "MIT"}`. This commit has no separate LICENSE file or
  explicit copyright notice; retain this provenance and verify the complete
  upstream licensing notices before redistributing the third-party source.
- Equivalent to trt-sam3 `src/common/ominicrop.hpp` at commit
  `593b4ae199c23e6a2042a7dff268c79a9904b253`, except for the constructor's
  default maximum crop size (1280 here versus 1008 there). The Ascend adapter
  always passes `max_size` and `padding` explicitly, so these defaults do not
  affect SAM3 requests.

The Ascend adapter in `src/common/refineCrop.hpp` adds input validation and
request budgets outside the upstream algorithm. CPU/OpenCV ROI extraction,
ACL inference, detection NMS and mask postprocessing remain Ascend-specific.
No CUDA, TensorRT, Python omnicrop package or runtime network fetch is required.

Placed under `src/third_party/` so Docker's business-code build stage copies it
without invalidating the tokenizers/SentencePiece/Abseil dependency cache.
