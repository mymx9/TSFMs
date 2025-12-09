# Time Series Foundation Models (TSFMs) Project

## Project Overview

This is a comprehensive Time Series Foundation Models (TSFMs) project that aggregates and provides access to multiple state-of-the-art time series forecasting models. The project contains three major time series foundation models:

1. **Chronos** - A family of models from Amazon/Amazon Science for time series forecasting based on language model architectures
2. **Moirai** - Universal Time Series Transformers from Salesforce/Salesforce AI Research 
3. **TimesFM** - A decoder-only foundation model from Google Research for time series forecasting

The project structure is organized with:
- `src/` - Source code for each of the three TSFM libraries
- `demo/` - Jupyter notebook examples for each model
- `pretrained/` - Local directory for storing model weights (pre-populated with models)
- `download.sh` - Script to download the pretrained models

## Project Components

### Chronos (Amazon)
- Located in `src/chronos-forecasting/`
- Includes Chronos-2 (latest version), Chronos-Bolt, and original Chronos models
- Supports univariate, multivariate, and covariate-informed forecasting tasks
- Offers both patch-based (Chronos-Bolt) and language model-based approaches
- Models range from 8M to 710M parameters

### Moirai (Salesforce)
- Located in `src/uni2ts/`
- Based on the Uni2TS library for unified training of universal time series transformers
- Includes Moirai-1.x, Moirai-MoE, and Moirai-2.0 models
- Uses the GluonTS framework for data handling
- Features mixture of experts (MoE) variants for improved performance
- Models available in small, base, and large sizes

### TimesFM (Google)
- Located in `src/timesfm/`
- Decoder-only architecture pretrained for time series forecasting
- TimesFM 2.5 is the latest version (200M parameters, 16k context length)
- Supports continuous quantile forecasting and covariate support via XReg
- Available in PyTorch and Flax implementations

## Pretrained Models

The project includes pre-downloaded models in the `pretrained/` directory:
- `pretrained/Salesforce/` - Moirai model variants
- `pretrained/amazon/` - Chronos model variants  
- `pretrained/google/` - TimesFM model variants

## Demos and Examples

Each model has dedicated demo notebooks in the `demo/` directory:
- `demo/chronos/` - Chronos-2 quickstart notebook with univariate and covariate forecasting examples
- `demo/Moirai/` - Multiple Moirai notebooks including forecasting with pandas and various model sizes
- `demo/timesfm/` - TimesFM example notebook with basic forecasting usage

## Building and Running

### Prerequisites
- Python 3.x environment with PyTorch and other dependencies
- Model weights can be downloaded using the included `download.sh` script

### Setup
1. The environment appears to already be set up with the models downloaded
2. Each model library can be used independently or together
3. Install dependencies specific to each model as needed

### Example Usage
From the demo notebooks:
- For Chronos: `pipeline = BaseChronosPipeline.from_pretrained("./pretrained/amazon/chronos-2", device_map="cuda")`
- For Moirai: `model = MoiraiForecast(module=MoiraiModule.from_pretrained("./pretrained/Salesforce/moirai-1.1-R-small"))`
- For TimesFM: `model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("./pretrained/google/timesfm-2.5-200m-pytorch")`

## Key Features

1. **Multi-Model Support**: Unified repository with three different state-of-the-art TSFMs
2. **Pretrained Weights**: Includes ready-to-use model weights for immediate experimentation
3. **Comprehensive Examples**: Extensive Jupyter notebook examples for each model
4. **Flexible Architecture**: Each model supports different approaches (language models, transformers, decoder-only)
5. **Covariate Support**: Many models support exogenous variables (covariates) for improved forecasting
6. **Probabilistic Forecasting**: All models provide probabilistic forecasts with quantiles
7. **Multiple Frameworks**: Uses PyTorch, HuggingFace, GluonTS, and other popular frameworks

## Project Purpose

This project serves as a unified research and experimentation environment for state-of-the-art Time Series Foundation Models. Researchers and practitioners can:
- Compare different TSFMs side-by-side
- Evaluate models on their own data
- Conduct ablation studies
- Perform zero-shot forecasting tasks
- Fine-tune models on specific datasets