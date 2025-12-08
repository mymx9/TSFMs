#!/bin/bash

pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

hf download Salesforce/moirai-1.0-R-small --local-dir ./pretrained/Salesforce/moirai-1.0-R-small
hf download Salesforce/moirai-1.1-R-small --local-dir ./pretrained/Salesforce/moirai-1.1-R-small
hf download Salesforce/moirai-2.0-R-small --local-dir ./pretrained/Salesforce/moirai-2.0-R-small
