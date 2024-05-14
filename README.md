This is the code for the review of the paper "Missing Data Imputation with Uncertainty-Driven Network"

### Folder Structure

    .
    ├── data                   # dataset files
    ├── baselines              # the baselines of GAIN, VGAIN and TDM
    ├── mechanism.py           # the missing mechanism
    ├── nngp.py                # the neural network gaussian process
    ├── data_loader.py         # data loader
    ├── utils.py               # some help functions
    ├── main_hnsw_fast.py         # the overall entrance
    ├── downstream_classification.py   # the downstream classification using SVM
    └── README.md

### Quick Start

```
bash run_MCAR.sh

bash run_MAR.sh

bash run_MNAR.sh
```

### Citation

```
@article{wang2024uncertainty,
  title={Missing Data Imputation with Uncertainty-Driven Network},
  author={Wang, Jianwei and Zhang, Ying and Wang, Kai and Lin, Xuemin and Zhang, Wenjie},
  journal={Proceedings of the ACM on Management of Data},
  volume={2},
  number={3},
  pages={1--25},
  year={2024},
  publisher={ACM New York, NY, USA}
}

```



