# CIFAR-10 Image‑Classification Pipeline  
_Group 3 — Saudi Digital Academy_

This project compares two approaches for classifying images from the CIFAR‑10 dataset:

1. **Custom Convolutional Neural Networks (CNNs)** – trained from scratch  
2. **Transfer Learning** – fine‑tuning a pre‑trained MobileNetV2 backbone

The best model is exposed through a lightweight Flask API so it can be consumed from any front‑end or downstream service.

| Model                       | Top‑1 Accuracy | Params   | Notes |
|-----------------------------|---------------:|----------|-------|
| `cnn_model_1` (simple)      | 0.82           | 5 M      | Baseline custom CNN |
| `cnn_model_2` (deeper)      | 0.86           | 9 M      | Saved as `cnn_model_2.h5` |
| **`tl_model` (MobileNetV2)**| **0.92**       | 2.3 M ✓  | Best overall – served by API |

<details>
<summary>Table of Contents</summary>

- [Project Structure](#project-structure)  
- [Quick Start](#quick-start)  
  - [1 · Set‑up](#1--set-up)  
  - [2 · Exploring the Notebooks](#2--exploring-the-notebooks)  
  - [3 · Running the Inference API](#3--running-the-inference-api)  
- [Training From Scratch](#training-from-scratch)  
- [Results & Visualisations](#results--visualisations)  
- [Team](#team)  
- [License](#license)  
</details>

---

## Project Structure
```text
.
├── G3_project_1_cnn.ipynb              # Builds & trains two vanilla CNNs
├── G3_project_1_tranferLearning.ipynb  # Fine‑tunes MobileNetV2
├── cnn_model_2.h5                      # Best handcrafted CNN weights
├── tl_model.h5                         # Final MobileNetV2 weights (best)
├── deploy.py                           # Flask + TensorFlow REST API
├── G3-project1-report.pdf              # Full technical report
├── Group-3-project-1presentation.pptx  # Slide‑deck summary
└── README.md                           # <––– you are here
```

> **Dataset** – CIFAR‑10 (60 k × 32 × 32 RGB) is automatically downloaded by TensorFlow/Keras inside each notebook.

---

## Quick Start

### 1 · Set‑up
```bash
# clone the repo
git clone https://github.com/<your-handle>/cifar10-cnn-tl.git
cd cifar10-cnn-tl

# create virtual environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
# OR, minimal:
pip install tensorflow==2.16.1 flask flask-cors pillow numpy jupyter matplotlib
```

### 2 · Exploring the Notebooks
```bash
jupyter lab
# open either notebook and run cells top-to-bottom
```
Both notebooks end by serialising their best checkpoints (`.h5`) into the repo.

### 3 · Running the Inference API
```bash
python deploy.py
# default: http://0.0.0.0:5000
```

Sample request:

```bash
curl -X POST \
     -F "file=@./sample_images/dog.jpg" \
     http://localhost:5000/predict
```

JSON response
```json
{
  "predicted_class": "dog",
  "probabilities": {
    "airplane": 0.0002,
    "automobile": 0.0001,
    "...": "…",
    "truck": 0.0023
  }
}
```

`deploy.py` automatically loads **`tl_model.h5`** at start‑up and uses the following class order:

```python
['airplane','automobile','bird','cat','deer',
 'dog','frog','horse','ship','truck']
```

---

## Training From Scratch
If you wish to re‑train:

1. Open **`G3_project_1_cnn.ipynb`** to train two custom CNNs.  
2. Open **`G3_project_1_tranferLearning.ipynb`** to fine‑tune MobileNetV2.

Both notebooks:

- Perform standardisation & data‑augmentation  
- Implement early‑stopping and learning‑rate scheduling  
- Save the best weights to `/` as `.h5`  

GPU acceleration (CUDA / Apple‑silicon / Colab) is highly recommended.

---

## Results & Visualisations
The report and slides contain:

- Training & validation accuracy / loss curves  
- Confusion matrices for all three models  
- Side‑by‑side comparison plots  
- Discussion of over‑fitting, LR schedules, and augmentation strategies  

See **`G3-project1-report.pdf`** for the full write‑up and references.

---

## Team
| Name (alphabetical)    | Role |
|------------------------|------|
| **Abdulaziz Alshehri** | Data preprocessing, CNN v2 |
| **Essa Alhassan**      | Transfer‑learning notebook |
| **Malak Alshaikh**     | Metrics & visualisation |
| **Razan Khalil**       | Flask deployment & docs |

> Undertaken for **Saudi Digital Academy – Deep Learning Project #2**  
> _Start: 04 Mar 2025 · Finish: 07 Mar 2025_

---

## License
This repository is released under the **MIT License** – see `LICENSE` file for details.

---

### عربي (اختياري)
> يقدّم المشروع نموذجين لتصنيف صور ‎CIFAR‑10‎ باستخدام **شبكات الالتفاف العصبية (CNN)** وتقنية **التعلّم بالنقل (Transfer Learning)**. أفضل نموذج (MobileNetV2) متاح عبر واجهة **Flask API** لاستقبال صورة وإرجاع الفئة المتوقعة.

---

Happy coding 🚀
