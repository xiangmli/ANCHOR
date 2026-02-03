# ANCHOR

## 🛠️ Environment Requirements

This project requires **Python 3.10**.

### Dependencies
Please ensure the following libraries are installed:

* `torch == 2.1.0`
* `pandas == 2.0.3`
* `scikit-learn == 1.3.2`
* `numpy == 1.24.4`

## 🚀 Usage

### 1. Noise Creation

To start the noise creation process:

```
cd agents
python adjudicator.py
```

### 2. Train Noise Recognizer

To train the noise discriminator (recognizer):

```
cd noise_discriminator/code
python main.py
```

### 3. Adversarial Training

To start the adversarial training process:

```
python adversarial_training.py
```