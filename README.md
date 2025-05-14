# ⚡ Convolutional Spiking Neural Network (CSNN) using snnTorch

This repository provides a detailed implementation of a **Convolutional Spiking Neural Network (CSNN)** trained on the MNIST dataset using biologically-inspired **Leaky Integrate-and-Fire (LIF)** spiking neurons. Built with [snnTorch](https://github.com/jeshraghian/snntorch) and PyTorch, this project demonstrates how spiking neuron models can be trained using surrogate gradients and used effectively for image classification.

Rather than computing outputs in a single forward pass like traditional deep networks, this model processes each input over **100 discrete time steps**, allowing neurons to integrate temporal input and emit binary spikes when membrane thresholds are crossed — mimicking biological computation in the brain.

---

## 🧠 Overview

Spiking Neural Networks (SNNs) represent the third generation of neural networks. They leverage **temporal coding**, **event-based computation**, and **sparsity**, enabling energy-efficient and brain-like behavior. In this project, we develop and train a CSNN that processes inputs dynamically across time steps, using a temporal window and LIF neurons to accumulate input signals and generate spike outputs. 

Training is achieved using surrogate gradients, which approximate the non-differentiable step function in backpropagation. This approach makes SNNs compatible with GPU-based deep learning frameworks.

---

## 🏗️ Architecture

This CSNN is composed of stacked convolutional layers for spatial feature extraction, followed by fully connected layers for decision making. Each convolutional block is followed by a LIF spiking neuron layer, which processes the output across time steps.

### 🔍 Layer-by-layer breakdown:

```
Input: [1 x 28 x 28] grayscale image

→ Conv2D(1, 32, kernel_size=3, stride=1, padding=1)         → [32 x 28 x 28]
→ MaxPool2D(kernel_size=2)                                  → [32 x 14 x 14]
→ LIF Spiking Layer

→ Conv2D(32, 64, kernel_size=3, stride=1, padding=1)        → [64 x 14 x 14]
→ MaxPool2D(kernel_size=2)                                  → [64 x 7 x 7]
→ LIF Spiking Layer

→ Flatten                                                    → [3136]
→ Linear(3136 → 512)
→ LIF Spiking Layer

→ Linear(512 → 10)
→ LIF Spiking Output Layer
```

- **Neuron Model**: Leaky Integrate-and-Fire (LIF)
- **Surrogate Function**: Fast Sigmoid with `slope=25`
- **Simulation Time**: 100 steps per image
- **Output Encoding**: Classification based on output spike count

---

## ⚙️ Implementation Details

| Component             | Value / Description                           |
|----------------------|------------------------------------------------|
| Framework            | PyTorch + snnTorch                            |
| Dataset              | MNIST (60k train / 10k test images)           |
| Neuron Model         | LIF (Leaky Integrate-and-Fire)                |
| Surrogate Gradient   | `fast_sigmoid(slope=25)`                      |
| Epochs               | 1 (adjustable for longer training)            |
| Batch Size           | 128                                            |
| Time Steps           | 100                                            |
| Optimizer            | Adam (lr = 0.01)                               |
| Loss Function        | `ce_rate_loss()` (cross-entropy on spike rate)|
| Evaluation Metric    | Accuracy from highest output spike count      |
| Device Support       | CUDA, Apple MPS, or CPU (auto-detected)       |

### ⏱️ Temporal Simulation

Each MNIST image is presented repeatedly across 100 simulation steps. At each step, neurons update their internal voltage state and emit spikes when the threshold is exceeded. The total number of spikes per output neuron is used to determine the predicted class.

---

## 🧪 Dataset: MNIST

- MNIST is a benchmark image dataset of handwritten digits (0–9)
- 28x28 grayscale images, 10 classes
- Preprocessing:
  - Resize to 28×28
  - Normalize to zero mean and unit variance
  - Convert to PyTorch tensors

---

## 📈 Results

After training the CSNN for 1 epoch on the MNIST dataset, the model achieved **state-of-the-art performance** for shallow spiking networks trained from scratch using surrogate gradients.

### ✅ Final Test Accuracy

```
Final Test Accuracy: 97.41%
```

### 📊 Performance Summary

- Accuracy evaluated using `accuracy_rate()` across all test samples
- Classification based on spike count across 100 time steps
- Model shows strong convergence even in a single epoch due to temporal integration and convolutional abstraction
- Loss decreases consistently; accuracy climbs rapidly and stabilizes above 97%

### 📉 Accuracy Progression

A real-time test accuracy plot is displayed at the end of training. The test accuracy rises steeply within the first few hundred iterations and plateaus around 97.4%, showing strong generalization and temporal learning capabilities.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/convolutional-spiking-neural-network.git
cd convolutional-spiking-neural-network
```

### 2. Install Dependencies

```bash
pip install torch torchvision matplotlib snntorch
```

### 3. Run the Model

```bash
python csnn_train.py
```

This will:
- Train the CSNN over 100 time steps per image
- Print loss and test accuracy every 50 steps
- Generate a final plot showing test accuracy over time

---

## 📁 Repository Structure

```
.
├── csnn_train.py          # Complete training and evaluation pipeline
├── README.md              # This documentation
├── LICENSE                # MIT License
```

---

## 📜 License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute this code with attribution.

See the [LICENSE](./LICENSE) file for full details.

---

## 🙏 Acknowledgements

This implementation is inspired by the paper:

> Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu.  
> *“Training Spiking Neural Networks Using Lessons From Deep Learning.”*  
> Proceedings of the IEEE, Vol. 111, No. 9, September 2023.  
> [DOI: 10.1109/JPROC.2023.3280284](https://doi.org/10.1109/JPROC.2023.3280284)

Special thanks to the developers of [snnTorch](https://github.com/jeshraghian/snntorch) for their excellent neuromorphic deep learning framework.

---

## 👤 Author

**Mani Majd**  
Engineering Science, University of Toronto  
📫 [LinkedIn](https://www.linkedin.com/in/mani-majd)  
💻 [GitHub](https://github.com/yourusername)

## 👤 Author

**Mani Majd**  
🔗 [LinkedIn](https://www.linkedin.com/in/mani-majd)  
