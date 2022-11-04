# KB-Recyclable-coffeecup-classification

## Quick Start
### Data preparation
In our experiments, we use custom dataset. The datasets should be put in data, respecting the following tree directory:
```
${ROOT}
|-- data
`-- |-- coffeecup
    `-- |-- train
        |   |-- plastic
        |   |-- paper
        |   |-- paper_in
        |   |-- waste
        `-- valid
            |-- plastic
            |-- paper
            |-- paper_in
            |-- waste
        `-- test
            |-- plastic
            |-- paper
            |-- paper_in
            |-- waste
```
### Training and Testing
For training:
```
python train.sh
```
For testing:
```
python infer.sh
```
### Model
 * Resnet [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)]
