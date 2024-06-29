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

#### 2. Download Dataset
```bash
cd data
kaggle competitions download -c lmsys-chatbot-arena
unzip lmsys-chatbot-arena.zip
```