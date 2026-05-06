# adapters/

Adapters are the stable boundary for resources that differ by source or format.

## Submodules

* `datasets`: dataset normalization adapters and category JSON helpers.
* `models`: model path profiles and TransformerLens loading helpers.
* `saes`: SAE path profiles and checkpoint format loaders.

Business scripts should select profiles and adapters by name instead of hard-coding
resource paths, checkpoint filenames, or raw dataset field mappings.
