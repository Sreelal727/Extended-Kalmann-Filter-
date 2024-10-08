
# Duty Cycle Prediction Using Artificial Neural Network (ANN)

This project involves building an Artificial Neural Network (ANN) for predicting the duty cycle based on voltage and current input. The trained ANN model is then used in an embedded system setup to regulate and predict duty cycles, which is transmitted over a network.

---

## Project Structure

- **Training Script (`ann_training.py`)**: This script preprocesses the dataset, trains the ANN model, and saves the model to an H5 file.
- **Testing and Deployment Script (`final.py`)**: This script loads the trained ANN model and uses it to predict the duty cycle from real-time voltage data, interacting with a server to send/receive data.

---

## Prerequisites

- Python 3.6 or later
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `tensorflow`
  - `scikit-learn`
  - `urllib`
  - `datetime`

You can install the required libraries using:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset (`data.csv`) includes columns for voltage (`v`), current (`i`), and the target output (`Target`). This data is used for training and testing the ANN model.

The features are normalized to improve the neural network's performance, and the dataset is split into training and testing sets.

---

## Model Training

The neural network has the following architecture:
- Input Layer: 1 input neuron (voltage)
- Hidden Layers: 4 dense layers with 64, 32, 16, and 8 neurons respectively, using ReLU activation.
- Output Layer: 1 neuron predicting the duty cycle.

The model is trained using `mean_squared_error` as the loss function and the Adam optimizer with a learning rate of 0.001. Early stopping is applied to prevent overfitting.

### Training the Model:

```bash
python ann_training.py
```

This script will:
- Load the dataset and preprocess it.
- Train the model with a validation split.
- Save the trained model as `duty_cycle_predictor_model.h5`.

---

## Real-Time Prediction and Communication

The script `final.py` loads the trained ANN model and performs the following tasks:
- Continuously requests real-time data (voltage and current) from the server using HTTP requests.
- Processes the data and uses the model to predict the duty cycle.
- Sends the predicted duty cycle back to the server for control applications.

### Running the Real-Time Prediction Script:

```bash
python final.py
```

This script will:
- Load the trained model (`duty_cycle_predictor_model.h5`).
- Continuously capture and process voltage data.
- Make predictions for the duty cycle.
- Transmit the duty cycle to the server via HTTP requests.

---

## Communication with ESP32

The script interacts with an ESP32 over HTTP using `urllib`. Data is sent and received over a local network, where the voltage and current values are passed to the model for duty cycle prediction, and the predicted duty cycle is transmitted back to control the hardware.

---

## Future Enhancements

- **Feature Expansion**: Adding more input features (like temperature, pressure, etc.) for more robust duty cycle prediction.
- **Model Optimization**: Experimenting with different neural network architectures or using techniques like transfer learning for improved accuracy.
- **Hardware Extension**: Extending the system to integrate more sensors for a complete control loop in embedded systems.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

