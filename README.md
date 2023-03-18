# MetaRF: Differentiable Random Forest for Reaction Yield Prediction with a Few Trails
This is the implementation for our paper "MetaRF: Differentiable Random Forest for Reaction Yield Prediction with a Few Trails".

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
* Run ``$ data_preprocessing.py`` to preprocess the data. 
* This step includes random forest module and dimension-reduction module. 
* The original reaction and yield data in this paper is from [[1]](#1), [[2]](#2) and [[3]](#3).
* We also provide the data after preprocessing in ``/data``(Two datasets are too large for github and they can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1uDrZnTRI1s4AXaDu9xJOzgpysCz10B5g?usp=share_link)). 

## Model training
* Run ``$ train.py`` to perform meta-training and model saving.
* Our trained model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1O-1K9h3k77MbM6E-spZ0hW6AsLQOoObB?usp=share_link).

## Model fine-tuning and testing
* Run ``$ test.py`` to perform few-shot fine-tuning, dimension-reduction based sampling method and model evaluation.
* We use relative path in this repository. Please place the downloaded model in the ``/model`` folder.
* [update] We add more baseline comparision in ``$ baseline.py``, including [RXNFP](https://github.com/rxn4chemistry/rxn_yields) [4], [DRFP](https://github.com/reymond-group/drfp) [5], etc.

## Tutorial
We provide a step-by-step tutorial that includes the whole workflow (including Data preprocessing, Model training, Model fine-tuning and testing, Baseline comparision) in ``$ Workflow of MetaRF - Tutorial.ipynb``. We also provide a colab version, which can help users easily access our code and environment by clicking: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HywJ1we0FU6imcR-SDABTo91SP1NUc7Y?usp=share_link)

```
Note: 
In this tutorial, we take the procedure for Buchwald Hartwig HTE dataset as an example. The other two datasets share the same procedure.
```

## Questions
For further question about the code, please contact 'kexinchen0526@gmail.com'.

## References
<a id="1">[1]</a> Ahneman, D.T., Estrada, J.G., Lin, S., Dreher, S.D., Doyle, A.G.: Predicting reaction performance in c–n cross-coupling using machine learning. Science 360(6385), 186–190(2018).

<a id="2">[2]</a> Perera, D., Tucker, J.W., Brahmbhatt, S., Helal, C.J., Chong, A., Farrell, W., Richardson, P., Sach, N.W.: A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science 359(6374), 429–434(2018).

<a id="3">[3]</a> Saebi, M., Nan, B., Herr, J., Wahlers, J., Guo, Z., Zura  ́nski, A., Kogej, T., Norrby, P.-O., Doyle, A., Wiest, O., et al.: On the use of real-world datasets for reaction yield prediction. ChemRxiv (2021).

<a id="4">[4]</a> Schwaller, P., Vaucher, A. C., Laino, T., & Reymond, J. L. (2021). Prediction of chemical reaction yields using deep learning. Machine learning: science and technology, 2(1), 015016.

<a id="5">[5]</a> Probst, D., Schwaller, P., & Reymond, J. L. (2022). Reaction classification and yield prediction using the differential reaction fingerprint DRFP. Digital discovery, 1(2), 91-97.

