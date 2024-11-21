# Faux-Hate Detection and Analysis - ICON 2024 Shared Task

This repository contains our work for the **ICON 2024 Shared Task on Decoding Fake Narratives in Spreading Hateful Stories (Faux-Hate)**. The task aims to address the detection of hate speech in social media comments, specifically focusing on the intersection of fake narratives and hate speech in code-mixed Hindi-English texts.

## Overview

The **Faux-Hate** task is designed to challenge participants in two main areas:

1. **Binary Faux-Hate Detection**: Identify if a social media comment contains fake information and if it contains hate speech.
2. **Target and Severity Prediction**: Predict the target (individual, organization, religion) and severity (low, medium, high) of hate speech in a given comment.

This repository includes models, data preprocessing pipelines, and evaluation strategies we used to tackle these tasks.


## Task Details

### Task A - Binary Faux-Hate Detection

In Task A, the objective is to identify whether a social media comment contains both fake information and hate speech. This involves:

- **Fake Label**: Whether the content is fake (1) or real (0).
- **Hate Label**: Whether the content contains hate speech (1) or not (0).

Participants are expected to develop a model capable of predicting both labels from the text samples.

### Task B - Target and Severity Prediction

In Task B, the objective is to classify the target and severity of hate speech in the text. The labels to predict are:

- **Target**: The intended target of the hate speech, which could be:
  - Individual (I)
  - Organization (O)
  - Religion (R)
  
- **Severity**: The severity of the hate speech, which could be:
  - Low (L)
  - Medium (M)
  - High (H)

The task requires a model that predicts both the target and the severity labels for each given text sample.

## Approach

We tackled the problem using **transformer-based models** like **XLM-R**, **mBERT** and **hateBERT**, fine-tuned on the dataset provided for both tasks.

## Requirements

To run the code in this repository, you need the following Python libraries:

- `numpy`
- `pandas`
- `torch`
- `transformers`
- `sklearn`
- `matplotlib`
- `seaborn`

You can install all required dependencies by running:

```bash
pip install -r requirements.txt
```

# Citation

```bibtex
@article{biradar2024faux,
  title={Faux Hate: Unravelling the Web of Fake Narratives in Spreading Hateful Stories: 
         A Multi-Label and Multi-Class Dataset in Cross-Lingual Hindi-English Code-Mixed Text},
  author={Biradar, Shankar and Saumya, Sunil and Chauhan, Arun},
  journal={Language Resources and Evaluation},
  pages={1--32},
  year={2024},
  publisher={Springer}
}
```
# Acknowledgements
- The ICON 2024 organizers for hosting this shared task.
- Hugging Face for providing powerful transformer-based models.
- The open-source community for providing tools and libraries that facilitated our work.