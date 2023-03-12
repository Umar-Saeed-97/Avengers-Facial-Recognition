# Avengers Facial Recognition

Avengers Face Recognition is an exciting computer vision project that aims to recognize and classify 5 popular Avengers actors - Robert Downey Jr. (Iron Man), Chris Evans (Captain America), Chris Hemsworth (Thor), Scarlett Johansson (Black Widow), and Mark Ruffalo (Hulk). The project utilizes deep learning algorithms to accurately classify images of the actors and provide a comprehensive understanding of each actor's visual characteristics. With this model, we aim to provide an engaging and fun way for Marvel fans to test their knowledge of the iconic characters and contribute to the development of facial recognition technology. Additionally, this project can be used in various applications such as security systems and marketing research. Whether you're a Marvel fan or just curious about facial recognition technology, this project is sure to capture your attention.

## Screenshots

![Prediction Result Snippet](https://drive.google.com/file/d/1j0rU2yHjp_MAMHvMgrtBJT9Sl3JUAS-5/view?usp=sharing)


## Requirements

The following packages are required to run the notebook:

- matplotlib
- numpy
- tensorflow
- pandas
- opencv-python
- seaborn
- scikit-learn
- deepface

You can install these packages using the following pip command:

```bash
    pip install -r requirements.txt
```

Note: The requirements.txt file contains all of the required packages and their respective versions. You can use this file to install the packages using the command above.

## Usage

To use this project, you can run the Project Notebook.ipynb notebook file in Jupyter Notebook or JupyterLab. This notebook contains all of the code used in the project, as well as detailed explanations of each step in the process.

The notebook loads the preprocessed dataset and performs face recognition on the images using the pre-trained model. It evaluates the model's performance using metrics such as accuracy and displays the results in visualizations such as a bar chart and a confusion matrix.

## Dataset

The dataset used in this project is the Avengers Faces Dataset, consisting of images of the five main actors from the Marvel Cinematic Universe (MCU) - Chris Evans, Chris Hemsworth, Mark Ruffalo, Robert Downey Jr, and Scarlett Johansson. The dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets/yasserh/avengers-faces-dataset).

The dataset contains 273 images, with each actor having approximately 50-60 images. These images were obtained from various movies and promotional events. Please review and abide by the terms of use and license information for the dataset before using it in your own projects.

## Model

The model used in this project is based on the DeepFace facial recognition model, with the ArcFace model as the base model. The model was used to compare the input image with images in the database and return the closest match, corresponding to the actor whose face appears in the input image.

The model was compiled with the euclidean_l2 distance metric, which measures the distance between the input image and the images in the database, and returns the closest match. The model was also set to enforce_detection=False, which allows it to make predictions even when faces are not detected in the input image.

Note that this project did not involve training a machine learning model, as the facial recognition model was pre-trained and simply used for inference.

## Evaluation

The model was evaluated on a test dataset which was approximately 18% of the original dataset. The model achieved an impressive accuracy score of 98% on the test dataset. A classification report and a confusion matrix were also generated to provide a more comprehensive understanding of the model's performance. The classification report displayed the precision, recall and f1-score for each of the Avenger class. The confusion matrix provided a graphical representation of the number of correct and incorrect predictions made by the model. Both the classification report and the confusion matrix were plotted to visualize the results and to gain further insights into the model's performance.

# License

MIT

## Author

[Umar Saeed](https://www.linkedin.com/in/umar-saeed-16863a21b/)


