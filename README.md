# ChocolateML

## Installation

To install the dependencies needed, navigate to the project folder and run the following command:
```bash
pip install -r requirements.txt
```
## Classification without manual labelling

1. Place the _results folder containing the dataset in the project folder.
2. Open the file `chocolatclass.ipynb`.
3. Execute the cells in the notebook one by one.
4. The models will be saved to 'models' folder.

## Classification(Multi label classification) for manually labelled images

1. Place the the 15 labelled image group folders in 'results' folder.
2. Place the 15 label csv files in 'labels' folder.
3. Place the images for inference in 'predict_inputs' folder.
5. Open the file `chocolatclass.ipynb`.
6. Execute the cells in the notebook one by one.
7. The models will be saved to 'classification_models' folder.
8. The inference results will be saved to 'predictions.csv'

## Author

Thomas Manjooran
