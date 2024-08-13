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
export HF_TOKEN="your_hf_token"
```

#### 2. Install unzip

```bash
sudo apt install unzip
```

#### 3. Download Datasets
```bash
kaggle datasets download -d lizhecheng/lmsys-datasets
unzip lmsys-datasets.zip
```

#### 4. Download LoRA Adapters
```bash
kaggle datasets download -d lizhecheng/lmsys-lora
unzip lmsys-lora.zip
```

### Training

#### 1. In this repo
```bash
cd src
cd team gemma / cd team llama
python train_xxx.py
```
#### 2. Go to the full repo
Click [full-training-code](https://github.com/2200xiaohu/LMSYS)


## [38th Solution] Lost Gold Medal

### 1. Code

Check our code at [LMSYS GitHub](https://github.com/Lizhecheng02/Kaggle-LMSYS).

### 2. Methodology

We employ instruction tuning, making the input format crucial. After experimenting with various formats, we identified the optimal approach:

First, we define a maximum length. Then, we concatenate multiple turns of prompt-response pairs within this limit. If a previous prompt-response pair exceeds the maximum length, the new prompt-response pair is placed in a separate row. For example, consider prompts [P1, P2, P3] with corresponding responses [A1, A2, A3] and [B1, B2, B3]. This method allows us to generate two rows: (P1, A1, B1) and (P2, A2, B2, P3, A3, B3), assuming (P1, A1, B1) does not exceed the maximum length. However, for training, we only use the last turn of the prompt-response pair for each ID.

This approach offers two key advantages:
1. Structuring the input in this way may help the model learn which two responses need to be compared.
2. Concatenating prompt-response pairs within the maximum length ensures that each input is a complete conversation, avoiding truncation. This reduces the risk of the model making bad choices due to incomplete responses.

```
<start_of_turn>user
Here are two question-answering dialogues. Compare two models' performance on answering questions, determine which is better.
#Prompt1
xxxxx
#Response
##Model A
xxxxx
##Model B
xxxx

#Prompt2
xxxxx
#Response
............

###options
A. Model A
B. Model B
C. Tie
<end_of_turn>
<start_of_turn>model 
A<eos>
```

### 3. Training Details
- 4bit QLoRA on [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) and [
Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), parameters: r = 32, modules = ["q_proj", "k_proj", "v_proj", "o_proj"].

- Instruction-tuning instead of classification.
- No gradient_checkpointing_enable() to reduce the training time.
- Used additional 33k data for fine-tuning and sample 10k data to do TTA.
- Great CV split (80% / 20%) to avoid duplicates between train and validation.
- GPU: multiple 80GB A100 GPUs + multiple A40 GPUs.

### 4. Not Work
- Pseudo-label and trained by hard label. (Maybe should consider use KL-loss to use pseudo-label)
- Only calculate [A, B, C] token loss even doing instruction-tuning, the same as classification task.

### 5. Conclusion
Because of some malicious people, what was once a very stable and meaningful competition has turned into one of the worst in Kaggle's history.

**Thanks to the entire team for everyone's hard work. Let's keep moving forward!**