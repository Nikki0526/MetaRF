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

## Data preprocessing
* Run ``$ data_preprocessing.py`` to preprocess the data and get prepared for the model training. This step includes random forest module and dimension-reduction module. 
* The original data in this paper is from [[1]](#1), [[2]](#2) and [[3]](#3).
* We also provide the data after preprocessing in ``/data``.

## References
<a id="1">[1]</a> Ahneman, D.T., Estrada, J.G., Lin, S., Dreher, S.D., Doyle, A.G.: Predicting reaction performance in c–n cross-coupling using machine learning. Science 360(6385), 186–190(2018).

<a id="2">[2]</a> Perera, D., Tucker, J.W., Brahmbhatt, S., Helal, C.J., Chong, A., Farrell, W., Richardson, P., Sach, N.W.: A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science 359(6374), 429–434(2018).

<a id="3">[3]</a> Saebi, M., Nan, B., Herr, J., Wahlers, J., Guo, Z., Zura  ́nski, A., Kogej, T., Norrby, P.-O., Doyle, A., Wiest, O., et al.: On the use of real-world datasets for reaction yield prediction. ChemRxiv (2021).
