# MetaRF: Differentiable Random Forest for Reaction Yield Prediction with a Few Trails

## Workflow
![image](https://raw.githubusercontent.com/Nikki0526/MetaRF/main/image/workflow_updated.png)

## Setup
1. Check dependencies
```
 - tensorflow==2.9.2
 - kennard-stone==1.1.2
 - numpy
 - pandas
 - sklearn
```
2. Clone this repo
```
git clone https://github.com/Nikki0526/MetaRF.git
```

## Preprocessing
* Run ``$ data_preprocessing.py`` to preprocess the data and get prepared for the model training. This step includes random forest module and dimension-reduction module. We also provide the processed data in ``/data``
