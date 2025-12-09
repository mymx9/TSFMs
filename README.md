# Time Series Foundation Models (TSFMs)

This repository contains a comprehensive collection of state-of-the-art Time Series Foundation Models (TSFMs), including Chronos (Amazon), Moirai (Salesforce), and TimesFM (Google), with pre-trained models and examples ready for experimentation.

## 🌟 Overview

Time Series Foundation Models (TSFMs) represents the cutting edge in time series forecasting, leveraging large-scale pre-trained models to provide accurate zero-shot and few-shot forecasting capabilities. This repository unifies three leading TSFM approaches:

- **Chronos** - Amazon's language model approach to time series forecasting
- **Moirai** - Salesforce's universal time series transformers
- **TimesFM** - Google's decoder-only foundation model for time series

## 📁 Project Structure

```
TSFMs/
├── demo/                  # Jupyter notebooks with examples for each model
│   ├── chronos/           # Chronos examples and quickstart guides
│   ├── Moirai/            # Moirai forecasting notebooks
│   └── timesfm/           # TimesFM examples
├── pretrained/            # Pre-trained model weights
│   ├── amazon/            # Chronos models
│   ├── Salesforce/        # Moirai models  
│   └── google/            # TimesFM models
├── src/                   # Source code for each TSFM library
│   ├── chronos-forecasting/ # Chronos source
│   ├── timesfm/           # TimesFM source
│   └── uni2ts/            # Moirai/Uni2TS source
└── download.sh            # Script to download additional model weights
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch environment with CUDA support (recommended for GPU acceleration)
- Model weights (pre-downloaded in the `pretrained/` directory)

### Setup

1. Clone and navigate to the repository:
```bash
cd /path/to/TSFMs
```

2. The models are pre-downloaded, but if you need to refresh them:
```bash
bash download.sh
```

### Basic Usage

#### Chronos-2 Example
```python
from chronos import BaseChronosPipeline
import pandas as pd

# Load the Chronos-2 pipeline
pipeline = BaseChronosPipeline.from_pretrained("./pretrained/amazon/chronos-2", device_map="cuda")

# Load and forecast your time series
context_df = pd.read_csv("your_timeseries_data.csv")
pred_df = pipeline.predict_df(context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
```

#### Moirai Example
```python
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("./pretrained/Salesforce/moirai-1.1-R-small"),
    prediction_length=24,
    context_length=512,
    patch_size="auto",
    num_samples=100,
    target_dim=1
)
predictor = model.create_predictor(batch_size=32)
```

#### TimesFM Example
```python
import timesfm
import numpy as np

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("./pretrained/google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
    )
)

point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[np.linspace(0, 1, 100)],
)
```

## 🧠 Models Overview

### Chronos (Amazon)
- **Architecture**: Language model-based or patch-based (Chronos-Bolt)
- **Capabilities**: Univariate, multivariate, and covariate-informed forecasting
- **Versions**: Chronos-2 (latest), Chronos-Bolt, original Chronos
- **Model Sizes**: 8M to 710M parameters
- **Features**: Zero-shot forecasting, state-of-the-art performance on multiple benchmarks

### Moirai (Salesforce)
- **Architecture**: Universal time series transformers
- **Capabilities**: Probabilistic forecasting with transformers
- **Versions**: Moirai-1.x, Moirai-MoE (Mixture of Experts), Moirai-2.0
- **Model Sizes**: Small, base, and large variants
- **Features**: Unified training framework, GluonTS integration, fine-tuning support

### TimesFM (Google)
- **Architecture**: Decoder-only transformer
- **Capabilities**: Time series forecasting with continuous quantiles
- **Versions**: TimesFM 2.5 (latest), TimesFM 1.0/2.0 (in v1 directory)
- **Model Sizes**: 200M parameters (TimesFM 2.5)
- **Features**: Continuous quantile forecasting, up to 16k context length

## 📊 Demos and Examples

Explore the `demo/` directory for comprehensive examples:

- **Chronos**: Check `demo/chronos/chronos-2-quickstart.ipynb` for univariate and covariate forecasting examples
- **Moirai**: Browse `demo/Moirai/example/` for various forecasting scenarios with different model sizes
- **TimesFM**: Review `demo/timesfm/example.ipynb` for basic usage patterns

## 📦 Dependencies

The environment should already have:
- PyTorch with CUDA support
- HuggingFace libraries
- GluonTS for Moirai
- Pandas and NumPy for data handling
- Matplotlib for visualization

## 🏷️ License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This project is primarily a unified access point for the three TSFM libraries. For contributions to the specific models, please refer to the original repositories:

- Chronos: https://github.com/amazon-science/chronos-forecasting
- Moirai: https://github.com/SalesforceAIResearch/uni2ts
- TimesFM: https://github.com/google-research/timesfm

## 📚 Citations

If you use these models in your research, please cite the original papers:

**Chronos:**
```
@article{ansari2024chronos,
  title={Chronos: Learning the Language of Time Series},
  author={Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=gerNCVqqtR}
}
```

**Moirai:**
```
@inproceedings{woo2024moirai,
  title={Unified Training of Universal Time Series Forecasting Transformers},
  author={Woo, Gerald and Liu, Chenghao and Kumar, Akshat and Xiong, Caiming and Savarese, Silvio and Sahoo, Doyen},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

**TimesFM:**
```
@article{lim2024decoder,
  title={A decoder-only foundation model for time-series forecasting},
  author={Lim, Bryan and Arik, Sercan O and Loeff, Niklas and Park, Juil and Sivakumar, Koushik and Maddix, Danielle and Bohlke-Schneider, Michael and Wang, Yuyang and others},
  journal={International Conference on Machine Learning},
  year={2024}
}
```