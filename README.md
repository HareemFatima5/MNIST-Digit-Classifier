# MNIST Digit Classifier 

This project implements a Convolutional Neural Network (CNN) for classifying handwritten digits from the MNIST dataset using TensorFlow/Keras.
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9) each sized 28x28 pixels.

## Project Structure

- `MNSIT_Digit_Classifier.ipynb`: Jupyter notebook with the complete pipeline including data loading, preprocessing, CNN model building, training, evaluation, and prediction.
- Utilizes TensorFlow/Keras for modeling.
- Optional integration with Streamlit and Ngrok for deploying the model via web interface.

## Requirements

Install the required packages using pip:

```bash
! pip install tensorflow streamlit pyngrok
```

## Dataset

The MNIST dataset is loaded directly from `keras.datasets.mnist`. It includes:

- 60,000 training images
- 10,000 testing images
- 10 classes (digits 0 through 9)

## Model Architecture

The CNN model architecture includes:

- Convolutional layers
- Max pooling layers
- Dropout for regularization
- Fully connected (Dense) layers
- ReLU activations and softmax for final classification

## How to Run

### Jupyter Notebook

Run the notebook directly using:

```bash
jupyter notebook MNIST_Digit_Classifier.ipynb
```

### Streamlit Deployment 

```bash
streamlit run app.py
```

If using Ngrok to expose the app publicly:

```bash
ngrok http 8501
```
## Streamlit App

![demo](https://github.com/HareemFatima5/Netflix-Movie-Recommender/blob/main/Netflix%20Movie%20Recommender.PNG)

## Results & Accuracy

The model achieved the following test accuracies:

| Training Epoch | Accuracy (%) |
|----------------|--------------|
| Final Model    | **97.99%**   |
| Other Runs     | 96.58%–98.96%|

- The highest observed test accuracy is **97.99%**
- Other experimental results range from **96.58% to 98.96%**
- These results show strong generalization performance for digit classification on MNIST

## Highlights

- Simple and effective CNN based solution
- High classification accuracy on a standard benchmark
- Easy to deploy using Streamlit and Ngrok

## License

This project is open source and available for educational and non commercial use.

