# DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/xxxx/blob/main/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2310.06367-B31B1B.svg)](https://arxiv.org/pdf/2310.06367.pdf)

<!-- [[Code](xxxx - Overview)] -->

![cover](framework.png)

Official code for the paper "DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening", accepted at *Neural Information Processing Systems, 2023*. **Currently the code is a raw version, will be updated ASAP**. If you have any inquiries, feel free to contact billgao0111@gmail.com

# Requirements

same as [Uni-Mol](https://github.com/dptech-corp/Uni-Mol/tree/main/unimol)

**rdkit version should be 2022.9.5**

### Inference without CUDA

Inference supports CPU and available PyTorch accelerator backends. Training is
unchanged. The following installation uses Python 3.10, CPU-only PyTorch, and
UniCore without compiling CUDA extensions:

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python -m pip install --no-build-isolation --no-deps "git+https://github.com/dptech-corp/Uni-Core.git@ace6fae1c8479a9751f2bb1e1d6e4047427bc134"
```

This UniCore revision defaults to disabling CUDA extensions; do not add
`--enable-cuda-ext`. Its layer normalization and softmax implementations fall
back to ordinary PyTorch operations. No CUDA toolkit is needed. For accelerators,
install the appropriate PyTorch build and any vendor backend package instead of
the CPU-only PyTorch command above. Optional vendor backends must be registered
with PyTorch before device selection; inference does not install or import them
automatically.

Download a checkpoint and input LMDB files as described below. From the repository
root, either edit the paths in the launchers or override them on the command line:

```bash
bash retrieval.sh --device cpu --path checkpoint_best.pt --mol-path mols.lmdb --pocket-path pocket.lmdb --emb-dir ./data/emb
bash retrieval.sh --device cuda:0 --path checkpoint_best.pt
bash test.sh --device cpu --test-task DUDE --path checkpoint_best.pt
```

Both Python entry points (`unimol/retrieval.py` and `unimol/test.py`) accept the
same device options:

| Argument | Behavior |
| --- | --- |
| omitted or `--device auto` | First available backend: CUDA/ROCm, XPU, MPS, other registered device modules alphabetically, then CPU. |
| `--device cpu` or `--cpu` | Force CPU, even when an accelerator is available. |
| `--device cuda:0`, `xpu:0`, `mps`, etc. | Require that device; an invalid or unavailable device is an error. ROCm uses PyTorch's `cuda` spelling. |
| `--device-id N` | Select index N when the device string omits an index; an explicit `--device TYPE:N` takes precedence. Default: 0. |

`--cpu` conflicts with an explicit non-CPU `--device`. Automatic selection checks
availability and device counts, not whether a backend implements every DrugCLIP
operator. Backends are discovered from installed PyTorch device modules exposing
`is_available()` (including the older `torch.backends.mps` probe); `cpu`, `meta`,
and `lazy` are not accelerator candidates. If the selected device later fails
during inference, the error is reported without retrying another device.

The effective device and precision are logged at startup. Inference runs in a
single process with gradients disabled and FP32 by default. Optional `--fp16` is
accepted only for CUDA/ROCm; `--bf16` is not supported by these entry points.
Molecule embeddings remain cached as NumPy arrays and final scoring remains on
CPU. Use a fresh `--emb-dir` when comparing devices or precision modes so existing
cached embeddings do not affect the comparison.

The benchmark entry point retains the existing data layouts:
`./data/DUD-E/raw/all/<target>/mols.lmdb` and `pocket.lmdb`, or
`./data/lit_pcba/<target>/mols.lmdb` and `pockets.lmdb`.

To run the regression tests:

```bash
python -m pip install pytest
python -m pytest tests -q
```

The tests mock accelerator availability and exercise real CPU molecule/pocket
encoding, checkpoint loading, embedding-cache reuse, retrieval, and both benchmark
paths using synthetic LMDB fixtures and a small randomly initialized model. They
do not require the published checkpoint or accelerator hardware.

## Data and checkpoints

https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing

It currently includes the train data, the trained checkpoint and the test data for DUD-E



### Training data

The dataset for training is included in google drive: train_no_test_af.zip. It contains several files:

```

dick_pkt.txt: dictionary for pocket atom types

dict_mol.txt: dictionary for molecule atom types

train.lmdb: train dataset

valid.lmdb: validation dataset

```

Use py_scripts/lmdb_utils.py to read the lmdb file. The keys in the lmdb files and corresponding descriptions are shown below:

```

"atoms": "atom types for each atom in the ligand" 

"coordinates": "3D coordinates for each atom in the ligand generated by RDKit. Max number of conformations is 10"

"pocket_atoms": "atom types for each atom in the pocket"

"pocket_coordinates": "3D coordinates for each atom in the pocket"

"mol": "RDKit molecule object for the ligand"

"smi": "SMILES string for the ligand"

"pocket": "pdbid of the pocket",
```


The dataset is compiled from the PBDBind dataset, containing a combination of authentic protein-ligand complexes and those generated through HomoAug, a technique for augmenting data with homology-based transformations.


### Test data

#### DUD-E

```
DUD-E
├── gene id
│   ├── receptor.pdb
│   ├── crystal_ligand.mol2
│   ├── actives_final.ism
│   ├── decoys_final.ism
│   ├── mols.lmdb (containing all actives and decoys)
│   ├── pocket.lmdb

```

#### PCBA

```
lit_pcba
├── target name
│   ├── PDBID_protein.mol2
│   ├── PDBID_ligand.mol2
│   ├── actives.smi
│   ├── inactives.smi
│   ├── mols.lmdb (containing all actives and inactives)
│   ├── pocket.lmdb

```


### Data preprocessing

see py_scripts/write_dude_multi.py

## HomoAug

Please refer to HomoAug directory for details

## Train

bash drugclip.sh

## Test

bash test.sh


## Retrieval 

bash retrieval.sh

In the google drive folder, you can find example file for pocket.lmdb and mols.lmdb under retrieval dir.


## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{gao2023drugclip,
    author = {Gao, Bowen and Qiang, Bo and Tan, Haichuan and Jia, Yinjun and Ren, Minsi and Lu, Minsi and Liu, Jingjing and Ma, Wei-Ying and Lan, Yanyan},
    title = {DrugCLIP: Contrasive Protein-Molecule Representation Learning for Virtual Screening},
    booktitle = {NeurIPS 2023},
    year = {2023},
    url = {https://openreview.net/forum?id=lAbCgNcxm7},
}
```
