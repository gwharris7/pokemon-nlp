# Gotta Classify 'em All!

Alexander Bacro, Samuel Levine, Grant Harris

# Abstract

Predicting a Pokémon's type based on its Pokédex entry is a challenging single-label and multi-label text classification problem. This project explores various methods, including classical machine learning approaches, neural networks, and fine-tuned transformer models, to classify Pokémon into their respective types using descriptive flavor text from the games. 

Beyond just the mechanics, this problem is inherently fun: can we predict the types of newly leaked Pokémon before they're officially revealed? Can we take characters from TV shows, movies, or other media and classify them as if they were Pokémon? We aim to find the best approach while also exploring the creative and unexpected ways type classification can be applied beyond just the games. 

We achieve over 95% accuracy using our best-performing models for both primary-type and dual-type classification.

# Approach & Dataset

This project investigates natural language processing (NLP) techniques for predicting a Pokémon’s type based on textual descriptions from the Pokédex. Pokémon types (e.g., Fire, Water, Electric) influence their abilities and strengths, making type classification a fundamental aspect of Pokémon games. Our goal is to implement, evaluate, and compare multiple advanced NLP models to find the best approach for this task. 

Our dataset consists of all 1025 Pokémon, with text data gathered from [PokéAPI](https://pokeapi.co/). The Pokédex contains descriptive entries for each Pokémon across multiple game versions, often with variations in wording and phrasing. To create a comprehensive representation, we concatenate all unique English entries for each Pokémon into a single textual description. This ensures that our models have access to a more complete semantic profile of each Pokémon, rather than relying on a single game's entry. It's worth noting, however, that older Pokémon will have much longer concatenated descriptions as a result. Charizard (Fire/Flying) is one of the very first Pokémon to be released, for example, so we combine all of it's unique descriptions that have been used in the various Pokémon games (see image below). 

<img width="1158" alt="image" src="https://github.com/user-attachments/assets/a9419f4b-519f-4f7c-bb24-b6bcfbe54144" /> 

## Preprocessing

For preprocessing, we clean and normalize the Pokédex text, removing formatting artifacts, special characters, and Pokémon names to reduce data leakage. One challenge with this dataset is type imbalance, particularly for Flying-type Pokémon. This occurs because many Pokémon classified as Flying are actually Normal/Flying dual-types, meaning that the Normal type dominates in raw counts. By swapping the order of Normal/Flying Pokémon to Flying/Normal, we allow our models to better learn Flying-type characteristics. This adjustment still leaves the vast majority of Normal/Non-Flying Pokémon untouched. 

![download](https://github.com/user-attachments/assets/66522632-8cc1-4c1d-8675-6f08d1b6edfe)

```python
# Standardizing Normal/Flying Pokémon as Flying/Normal
df.loc[(df['type1']=='normal') & (df['type2']=='flying'), ['type1','type2']] = ['flying', 'normal']
```

![download (1)](https://github.com/user-attachments/assets/626e3fe3-7857-431c-806c-386608847ef1)

Our goal is to classify Pokémon based on their Pokédex flavor text, with an ~80% accuracy target for single-type classification. We set this target based on existing results we found online, where a BERT-based model achieved 84% accuracy. We explore various NLP-based feature extraction and classification models to identify the most effective approach. Our initial methods include one-hot encoding and GloVe embeddings, followed by BERT-based models, which show significant improvements. Our best-performing model so far is a fine-tuned BERT implementation, achieving ~95% accuracy for single-type classification, with expectations of further optimization.

A major breakthrough comes when we introduce **oversampling** to handle type imbalance, particularly for underrepresented types like Flying and Ice. We use `RandomOverSampler` to generate additional synthetic samples for these rare types, helping the model to better recognize and classify them. After implementing oversampling, we see a noticeable improvement in the model’s ability to predict less common types, reducing the misclassification rate and increasing overall performance across the board.

# Methods

The dataset outlined above was used for all of our experiments. 

## Primary-Type Classification

1. One-Hot Encoding + Logistic Regression
 - **Setup:** Each Pokédex entry was tokenized, stemmed, and converted into a binary one-hot encoding, representing the presence or absence of words. The resulting vectors were used as input to a Logistic Regression model for type classification.
 - **Hyperparameters:** Minimal customization was done in terms of hyperparameters. We used NLTK’s `word_tokenize` and `SnowballStemmer` for stopwords and stemming respectively. And we used `RandomOverSampler` from `imblearn` to help with balancing the class distributions. 
 - **Results & Observations:** With an accuracy of 48%, this model performed significantly better than random guessing. It still struggled with some more context-dependent type cues. A t-SNE visualization of one-hot encoded representations showed some clustering of similar types, but separability was weak. This model is concerned more about token presence as opposed to semantic relationships—but that’s not necessarily a bad thing in this specific exercise! 

![newplot](https://github.com/user-attachments/assets/257b0fb5-6b46-40c0-9047-d6ff8c7931ff)

2. TF-IDF
 - **Setup:** Each Pokédex entry was processed using TF-IDF vectorization, where word importance was determined based on term frequency-inverse document frequency scores. The top five highest-scoring words were retained as features for classification. Given the relatively small dataset size, we used a Support Vector Machine (SVM) classifier, which performs well in high-dimensional spaces. To assess the effect of TF-IDF filtering, we trained two models: one using only the top five words per entry and another using the full, unfiltered Pokédex entries.
 - **Hyperparameters:** Like above, minimal hyperparameter customization was done here. 
 - **Results & Observations:** The TF-IDF-filtered model achieved 33% accuracy, while the unfiltered model performed better at 44%, suggesting that retaining all words provides a stronger signal for classification. This indicates that TF-IDF filtering may inadvertently remove critical context, particularly in generations with fewer Pokédex entries per Pokémon. Future improvements may involve adjusting the number of retained words or using n-gram features to preserve more meaningful phrases.

3. GloVe Embeddings + Classifier
 - **Setup:** Each Pokédex entry was tokenized, stemmed, and mapped to pre-trained 100D GloVe word embeddings. For each entry, the mean of all token embeddings was used as its final representation. A Bernoulli Naïve Bayes classifier was trained on these averaged embeddings. 
 - **Hyperparameters:** The word embeddings are `glove.6B.100d` (Stanford NLP GloVe). All other hyperparameters are the same aside from changing the classifier to a Bernoulli Naive Bayes. 
 - **Results & Observations:** Accuracy dropped 16% compared to one-hot encoding (down to 32%), suggesting that pre-trained GloVe embeddings may lack domain-specific knowledge for Pokémon-related terminology. The model struggled with contextual clues, likely because GloVe embeddings are static and do not adapt to sentence-level meanings. 

![newplot (3)](https://github.com/user-attachments/assets/3eccefa5-5881-429d-a256-ab814ed003d9)

4. BERT Fine-Tuning
 - **Setup:** We fine-tuned a BERT-based transformer model (bert-base-uncased) to classify Pokémon types using Pokédex flavor text. Entries were tokenized with a custom PyTorch dataset class, ensuring a structured pipeline for training and evaluation.
 - **Hyperparameters:** For training, we used bert-base-uncased with a maximum sequence length of 128. We set the batch size to 16 for both training and evaluation. The learning rate was `5e-5`, optimized using AdamW with a weight decay of 0.01. We trained the model for 10 epochs, evaluating and saving the best model at the end of each epoch. Accuracy was used as our evaluation metric. 
 - **Results & Observations:** With an accuracy of 95%, BERT massively outperformed earlier methods, showing that contextual embeddings provide a stronger signal for Pokémon type classification. The structured dataset and training pipeline were more scalable and reproducible compared to previous experiments. Code refactoring efforts in this experiment led to a more modular and efficient approach, and we plan to refactor our earlier experiments to align with this methodology. 

![newplot (8)](https://github.com/user-attachments/assets/8b4ab094-ed3d-4c27-9c9d-2dcec97f3dab)

## Dual-Type Classification

1. BiLSTM
 - **Setup:** To investigate the model's ability to classify Pokémon by primary and secondary types, we implemented a bidirectional LSTM (BiLSTM) model with BERT embeddings for dual-type classification. The labels were transformed into one-hot vectors representing the primary and/or secondary types of each Pokémon. A thresholding approach was applied to the logits representing the model output, allowing the model to predict multiple types if the corresponding logits were above a certain threshold.
 - **Hyperparameters:** The BiLSTM classifier model was trained over 10 epochs with a batch size of 32, max sequence length of 128, and a learning rate of 0.001. Adam optimizer and a sigmoid activation function were used, and the loss was calculated using binary cross-entropy loss with logits. The thresholding value was fine-tuned to be 0.25 (only types with output probabilities over 0.25 were predicted).
 - **Results & Observations:** The model achieved 85% accuracy on the validation set, with a validation loss of 0.07 and a macro average F1-score of 0.90.

2. BERT (No Oversampling)
 - **Setup:** We fine-tuned a BERT-based transformer to perform multi-label classification of Pokémon types, using a custom PyTorch dataset that tokenized entries and generated multi-hot label vectors combining both primary and secondary types.
 - **Hyperparameters:** The model used a maximum sequence length of 128 with a batch size of 16, a learning rate of 5e-5 optimized via AdamW with 0.01 weight decay, and was trained for 10 epochs while saving the best model at each evaluation epoch.
 - **Results & Observations:** The model achieved a macro-average F1-score of 0.51, with strong performance on types like electric and fire; however, it struggled on underrepresented classes—most notably, the “fairy” type recorded 0 precision and recall. Fairy-type was granted retroactively to some Pokémon in a later generation, giving the model trouble in predicting using these Pokémon’s old Pokédex entries and motivating the need for oversampling techniques to better balance multi-type predictions.

3. BERT (Oversampling)
 - **Setup:** We fine-tuned a BERT-based transformer for multi-label Pokémon type classification from Pokédex text, applying oversampling on the training set to balance underrepresented types.
 - **Hyperparameters:** The model was trained using the same hyperparameters as the BERT single-type model with a maximum sequence length of 128 and a batch size of 16, using AdamW with a 5e-5 learning rate and 0.01 weight decay over 10 epochs, with the best model saved at each evaluation epoch. 
 - **Results & Observations:** Oversampling led to a substantial performance boost (macro-average F1-score of 0.97), with near-perfect precision and recall across all types—including previously underrepresented classes like "fairy"—demonstrating that balancing the dataset significantly improves multi-label prediction.

# Overview of Results  

Our experiments show that more sophisticated models with contextual embeddings significantly outperform traditional methods in Pokémon type classification. One-hot encoding and TF-IDF provide reasonable baselines, but pre-trained embeddings like GloVe underperform due to limited domain specificity. Fine-tuned BERT models demonstrate the strongest performance for both single-type and dual-type classification, with oversampling dramatically improving multi-label performance. Below is a summary of key results:

## **Single-Type Classification Results**  

| Model | Accuracy | F1-Score | Notes |  
|-------|----------|----------|-------|  
| **One-Hot + Logistic Regression** | 48% | 0.47 | Stronger than random guessing; weak on context-dependent types |  
| **TF-IDF (Filtered)** | 34% | 0.21 | Filtering top 5 words weakens signal |  
| **TF-IDF (Unfiltered)** | 46% | 0.42 | Retaining all words improves accuracy |  
| **GloVe + Naive Bayes** | 32% | 0.31 | Pre-trained embeddings struggle with domain-specific terms |  
| **BERT Fine-Tuning** | **95%** | **0.95** | Best-performing model; strong context understanding |  

## **Dual-Type Classification Results**  

| Model | Accuracy | F1-Score | Notes |  
|-------|----------|----------|-------|  
| **BiLSTM** | 85% | 0.90 | Strong sequence modeling, but weaker on rare types |  
| **BERT (No Oversampling)** | 64% | 0.51 | Struggles with underrepresented types (e.g., Fairy) |  
| **BERT (With Oversampling)** | **96%** | **0.97** | Oversampling leads to near-perfect prediction across all types |  

Fine-tuned BERT consistently delivers the highest accuracy, showing that contextual embeddings and balanced data are critical for successful Pokémon type classification. Oversampling proves essential for improving dual-type classification, particularly for rare types.  

# Conclusion  

This project demonstrates that natural language processing (NLP) techniques can effectively classify Pokémon types based on Pokédex flavor text, achieving strong results with fine-tuned transformer models. While classical models like one-hot encoding and TF-IDF provided reasonable baselines, their limitations in capturing contextual meaning prevented them from reaching high accuracy. Pre-trained embeddings like GloVe also struggled due to the specialized nature of Pokémon-related language. Fine-tuning BERT yielded a substantial leap in performance, with accuracy reaching **95%** for single-type classification and dual-type classification with oversampling. The success of BERT highlights the importance of contextual embeddings and balanced data in text classification tasks.

Beyond the core classification task, we applied our trained models for fun to classify well-known fictional characters like **Godzilla** (Dragon-type), **Batman** (Dark/Fighting-type), **Harry Potter** (Psychic-type), and **Ariel** from *The Little Mermaid* (Water-type). 

![batman](https://github.com/user-attachments/assets/6a3344ec-fc33-49a6-bfaf-eb98f28b34f9)
![godzilla](https://github.com/user-attachments/assets/d0cef995-3801-49af-8d67-1b27ab58593d)
![harry potter](https://github.com/user-attachments/assets/88cae68d-50c3-4df1-8e61-7e1238928b14)
![mermaid](https://github.com/user-attachments/assets/8231758a-3725-42cc-b750-89ba5bf7e269)

This project was awesome. It was both fun and we actually pulled it off! Thanks for reading!
