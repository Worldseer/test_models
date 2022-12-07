# AxialGO: Deep axial-attention network for protein function prediction from sequence

AxialGO+ is a protein function prediction model built by the [AxialNet](https://github.com/Worldseer/axial-deeplab) backbone network using only protein features. The model structure is shown in the figure below.
![AxialGO](https://github.com/Worldseer/AxialGO/blob/main/images/AxialGO.jpg)

## Dependencies

* The code was developed and tested using python 3.8.3.
* We provide the dependencies to install the conda environment, first you need to install [ananconda](https://docs.anaconda.com/anaconda/install/index.html) on your computer, and then install the dependencies use:

  ```conda create --name <env> --file requirements.txt```
or
  ```conda env create -f environment.yml```
* For the integration of AixalGO and DiamondScore to obtain AxialGO+, the [diamond](https://github.com/bbuchfink/diamond) package needs to be installed.



## Data

* data_cafa3/：The CAFA3 dataset we used, includes training, validation and test sets and  the go.obo file
* data_2016/：The 2016 dataset we used, includes training, validation and test sets and  the go.obo file

## Scripts
- train_axialgo.py：used to train axialgo and output prediction files
- generate_data_loader_all.py：generate six styles of embedded winding matrix, use the trainloader function in it to generate an iterable DataLoader. The DataLoader has two outputs which are X list of matrices containing six winding styles and y is the true label.
- axialnet.py: contains AxialNet backbone network code, used to build AxialGO
- resnet.py: contains the ResNet backbone network code
- googlenet.py: contains GoogLeNet backbone network code
- vgg.py: contains the VGG backbone network code
- alexnet.py: contains the alexnet backbone network code
- create_model.py: contains  code for building AxialGO and code for using other backbone network models
- utils.py: codes for Gene Ontology terms

## Trained model
* model/: Contains the parameters of the model trained in the CAFA3 and 2016 datasets. Both model parameters provided are trained using the winding style (a), winding matrix size of 40 and embedding dimension of 16 from the paper.

## Training model
- Training the model with default parameters:
You can train the model directly with the default parameters by running `python train_axialgo.py`. Line 65 in the train_axialgo.py file will print the loss values to test if the model is working properly. We recommend commenting out this line if everything works
- Training models with custom parameters,
Please use:
```
python train_axialgo.py --data-root ./data_2016 --epochs 100 --batch-size 16 --epochs 30 --emb-dim 16 --winding-size 40
```

## Evaluate prediction.pkl
- Use the following command to evaluate the resulting prediction.pkl
```
python evaluate_plus.py --train-data-file ./data_2016/train_data.pkl --test-data-file ./predict/prediction_2016.pkl --terms-file ./data_2016/terms.pkl --go-file ./data_2016/go.obo --diamond-scores-file ./data_2016/test_diamond.res --ont mf
```
 
