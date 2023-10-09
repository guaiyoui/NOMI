This is the code for the review of the paper "Uncertainty-Driven Network for Missing Data Imputation"

### Folder Structure

    .
    ├── data                   # dataset files
    ├── model.py               # network structure
    ├── layers.py              # layers in the net
    ├── mechanism.py           # the missing mechanism
    ├── nngp.py                # the neural network gaussian process
    ├── data_loader.py         # data loader
    ├── utils.py               # some help functions
    ├── main_hnsw_fast.py         # the overall entrance
    └── README.md

### Quick Start

bash run_MCAR.sh

bash run_MAR.sh

bash run_MNAR.sh



