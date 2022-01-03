# BoundaryNet2
An improved version of BoundaryNet designed for Indiscapes 2 dataset .

## Installation
If package manager [pip](https://pip.pypa.io/en/stable/) is preferred , then please use the following commands : 

```bash
pip install -r requirements.txt
```
An alternative installation could be via [conda](https://www.anaconda.com/) : 
```bash
conda env create --file configs/BoundaryNetStable.yml --python=3.8
conda activate bnet 
```

## Directory 
```
├── config                  # Stable .yml files 
├── docs                    # Documentation files 
├── src                     # Source Codes for MCNN & AGCN Train & Test
├── models                  # Model codes 
├── utilities               # Utils files for contourisation 
├── losses                  # Associated loss functions 
├── datasets                # Final corrected jsons 
├── bash scripts            # ADA specific bash files
├── LICENSE
└── README.md
```

## Usage

To either train/test/validate place the appropriate flag in the command . Please opt for -h (help) for detailed explaination of the arguments .Make sure the experiment JSON & weight files are placed in their respective folders . 

### MCNN : 
```bash
python3 mcnn.py --mode 'train' --expdir 'mcnn/' --exp 'mcnn_config.json' --training-type 'normal' 
```
### AGCN : 
```bash
python3 agcn.py --mode 'train' --expdir 'agcn/' --exp 'agcn_config.json' --training-type 'normal'  --model_weights 'mcnn_stable.pth'
```

**Before these operations, ensure the folder `datasets` is structured as follows : **
```
    datasets
    ├── train
    │   ├── train.json
    ├── val
    │   └── val.json 
    |── test
    │   └── test.json 
    ├── train-val
        └── train-val.json
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


