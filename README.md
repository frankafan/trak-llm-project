# Project Overview

Model Interpretation and Performance Improvement with Large Language Models and Data Attribution: For this project we develop methods to improve language model performance by using important data as context, and use large language models to better explain smaller models with data attribution results.

## File Descriptions

- **bert_analysis.ipynb**: Analyzes BERT model performance on a dataset.
- **bert_classification_trak.ipynb**: Implements BERT-based text classification with TRAK analysis.
- **bert_finetuned_classification_trak.ipynb**: Fine-tunes a BERT model for text classification and performs TRAK analysis.
- **bert_predictions.csv**: Contains predictions made by the BERT model.
- **bert_prompt_engineering.ipynb**: Applies prompt engineering techniques to improve BERT model performance.
- **bert_trak_scores.csv**: Contains TRAK scores for BERT model training examples.
- **cifar_clip.ipynb**: Implements image classification using the CLIP model on the CIFAR dataset.
- **cifar_llm.ipynb**: Uses a large language model for CIFAR image classification.
- **cifar_resnet.ipynb**: Implements ResNet model for CIFAR image classification.
- **cifar_trak.ipynb**: Performs TRAK analysis on CIFAR image classification.
- **gpt_qnli_predictions.csv**: Contains predictions made by the GPT model on the QNLI dataset.
- **image_info.npy**: Stores image information for the CIFAR dataset.
- **llm_info.npy**: Stores information for the large language model.
- **predictions.csv**: General predictions file.
- **qnli_gpt.ipynb**: Implements GPT model for QNLI question-answering tasks.
- **qnli_rag_full.ipynb**: Implements RAG (Retrieval-Augmented Generation) model for full QNLI tasks.

## Setup Instructions

1. **Clone the Repository**:

   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.10 or later installed. Install the required packages using pip:

   ```sh
   pip install -r requirements.txt
   ```

3. **Download Necessary Data**:
   Some notebooks may require downloading datasets. Follow the instructions within each notebook to download the necessary data.

## Running the Notebooks

1. **Launch Jupyter Notebook**:

   ```sh
   jupyter notebook
   ```

2. **Open the Desired Notebook**:
   Navigate to the notebook you want to run (e.g., [bert_classification_trak.ipynb](http://_vscodecontentref_/0)) and open it.

3. **Run the Notebook**:
   Follow the instructions within the notebook to execute the cells. Ensure you run the cells in order to avoid any errors.

## Notebooks to run to get the data in the report (Notebooks should be run in order of the list below)

### QNLI Task

- **bert_classification_trak.ipynb**:

  - Run this notebook to load a BERT model on the GLUE and QNLI dataset, then compute the TRAK scores for this model and save the TRAK scores in `bert_trak_scores.csv`.

- **bert_finetuned_classification_trak.ipynb**:

  - Does the same thing as `bert_classification_trak.ipynb`, but on a fine tuned BERT model.

- **bert_analysis.ipynb**:

  - Analyzes BERT model performance.
  - Run the fine tuned BERT model on the QNLI validation set to get the model's validation accuracy.
  - Store the model's predictions in `predictions.csv`.

- **qnli_gpt.ipynb**:

  - Perform the QNLI task with OpenAI's GPT model using a custom RAG model and the training set examples after data attribution.
  - Set the OpenAI API key in the `openai.api_key` variable.

### CIFAR Task

- **cifar_resnet.ipynb**:

  - Run this notebook to train a ResNet9 architecture on the CIFAR-10 dataset. We will use this model for data attribution with TRAK.
  - Save model checkpoints in the directory `CHECKPOINT_DIR`

- **cifar_trak.ipynb**:

  - Run this notebook to load the ResNet9 model and compute the TRAK scores of on the training set.
  - Load model checkpoints from the directory `CHECKPOINT_DIR`, and save the images with their TRAK scores in the directory `IMAGE_DIR`.

- **cifar_clip.ipynb**:

  - Run this notebook to run the CLIP model on the scored CIFAR-10 training data.
  - Load the images with their TRAK scores in the directory `IMAGE_DIR`.

- **cifar_llm.ipynb**:

  - Run this notebook to run the CLIP model then generate LLM descriptions and score the reconstruction accuracy.
  - Ensure you have the `torchvision` and `transformers` libraries installed.
  - Load the images with their TRAK scores in the directory `IMAGE_DIR`, and set the OpenAI API key in the `api_key` variable.
