# Third-party code

- `src/deit.py` adapts the [DeiT](https://github.com/facebookresearch/deit)
  models and timm vision-transformer components, under Apache-2.0
  (`LICENSES/deit.txt`). Local model variants and portability fixes modify the
  upstream implementation; the copyright notices are retained.
- `src/convnext.py` adapts [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
  under the MIT license (`LICENSES/convnext.txt`).
- timm, torchvision, PyTorch, and LPIPS are installed dependencies and retain
  their respective licenses. Samplers follow PyTorch's distributed samplers,
  as documented in their source.
