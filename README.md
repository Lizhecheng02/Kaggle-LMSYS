## This Repo is for [Kaggle - LMSYS - Chatbot Arena Human Preference Predictions](https://www.kaggle.com/competitions/lmsys-chatbot-arena)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Install unzip

```bash
sudo apt install unzip
```

#### 3. Download Datasets
```bash
cd data
kaggle datasets download -d lizhecheng/lmsys-datasets
unzip lmsys-datasets.zip
```

#### 4. Download LoRA Adapters
```bash
kaggle datasets download -d lizhecheng/lmsys-lora
unzip lmsys-lora.zip
```