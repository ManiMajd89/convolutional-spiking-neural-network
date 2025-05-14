# ⚡ Convolutional Spiking Neural Network (CSNN) using snnTorch

This repository contains an implementation of a **Convolutional Spiking Neural Network (CSNN)** for MNIST digit classification using [snnTorch](https://github.com/jeshraghian/snntorch) and PyTorch. The model simulates biologically-inspired **Leaky Integrate-and-Fire (LIF)** neurons, processing input images over 100 time steps using surrogate gradient learning. It is built to emulate the temporal and sparse nature of neural computation, offering a foundation for neuromorphic and event-based vision systems.

## 🧠 Model Architecture

This spiking neural network processes each input over multiple timesteps, allowing neurons to accumulate membrane potential and fire spikes when thresholds are crossed. The network includes two convolutional layers, each followed by max pooling and a LIF spiking layer, then two fully connected layers with spiking output.

```
Input: [1 x 28 x 28] grayscale image

→ Conv2D(1, 32, 3x3, padding=1)        → 28x28
→ MaxPool2D(2x2)                       → 14x14
→ LIF Spiking Layer

→ Conv2D(32, 64, 3x3, padding=1)       → 14x14
→ MaxPool2D(2x2)                       → 7x7
→ LIF Spiking Layer

→ Flatten                              → 3136
→ Linear(3136 → 512)
→ LIF Spiking Layer

→ Linear(512 → 10)
→ LIF Spiking Output Layer
```

## 🧪 Dataset

- **MNIST** handwritten digits (0–9)
- 60,000 training and 10,000 test samples
- Preprocessing:
  - Resize to 28x28
  - Grayscale conversion
  - Normalize to mean = 0, std = 1
  - Batch size = 128

## ⚙️ Training Configuration

- Epochs: 1  
- Batch Size: 128  
- Time Steps: 100  
- Learning Rate: 1e-2  
- Optimizer: Adam  
- Loss Function: `ce_rate_loss()` (cross-entropy based on spike count)  
- Evaluation: Accuracy based on spiking output neurons  
- Device: CUDA / MPS / CPU (auto-detected)

Test accuracy is evaluated every 50 training iterations, and a real-time accuracy plot is generated at the end of training.

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

### 3. Run the Training Script

```bash
python csnn_train.py
```

## 📊 Example Output

```
Iter 0, Test Acc: 10.42%
Iter 50, Test Acc: 84.53%
Iter 100, Test Acc: 89.12%
...
Final Test Accuracy: 90.37%
```

At the end, a matplotlib plot is displayed showing accuracy progression over the training process.

## 📂 Project Structure

```
.
├── csnn_train.py          # Training script
├── README.md              # Project documentation
├── LICENSE                # MIT License
```

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this code with attribution.  
See the [LICENSE](./LICENSE) file for full terms.

## 🙏 Acknowledgements

This project is inspired by the work:

> Jason K. Eshraghian, Max Ward, Emre Neftci, Xinxin Wang, Gregor Lenz, Girish Dwivedi, Mohammed Bennamoun, Doo Seok Jeong, and Wei D. Lu.  
> *“Training Spiking Neural Networks Using Lessons From Deep Learning.”*  
> Proceedings of the IEEE, Vol. 111, No. 9, September 2023.  
> DOI: [10.1109/JPROC.2023.3280284](https://doi.org/10.1109/JPROC.2023.3280284)

Thanks also to the creators of [snnTorch](https://github.com/jeshraghian/snntorch) for enabling accessible and powerful SNN development in PyTorch.

## 👤 Author

**Mani Majd**  
🔗 [LinkedIn](https://www.linkedin.com/in/mani-majd)  
