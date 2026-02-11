### Train dataset

1) Download CAD-Recode dataset [train / val](https://huggingface.co/datasets/filapro/cad-recode-v1.5)
2) Convert CadQuery programs to meshes with *cadrecode2mesh.py* script
3) Split dataset in three groups with *cadrecode_split.py* script

The structure should be as follows:
```
cad-recode-v1.5-3groups
    ├── 0
    │   ├── train
    │   │   ├── 0.py
    │   │   ├── 0.stl
    │   │   └── ...
    │   └── val
    │       ├── 0.py
    │       ├── 0.stl
    │       └── ...
    ├── 1
    │   ├── train
    │   └── val
    └── 2
        ├── train
        └── val
```