## `CADReasoner`: Iterative Program Editing for CAD Reverse Engineering

 
This repository contains an implementation of the method introduced in our paper:

> **CADReasoner: Iterative Program Editing for CAD Reverse Engineering**<br>
> Soslan Kabisov
> Vsevolod Kirichuk
> Andrey Volkov
> Gennadii Savrasov
> Marina Barannikov
> Anton Konushin
> Andrey Kuznetsov
> Dmitrii Zhemchuzhnikov

### Installation

Install Python packages according to [Dockerfile](Dockerfile). Download and preprocess data following [instruction](data/README.md).

### Train

To launch training curriculum run *training_curriculum.sh* script:
```shell
./training_curriculum.sh <train_dataset> <per_device_train_batch_size> <n_gpus>
```

### Inference

To predict CadQuery codes run:
```shell
python3 test.py --dataset <test_dataset> --checkpoint <checkpoint> --n_iters <n_iters> --outdir <outdir> 
```

### Evaluation

To evaluate metrics run:
```shell
python3 evaluate.py --dataset <test_dataset> --pred_dir <pred_dir> 
```



### Citation
