# NLP Exercise 2: Aspect-Based Sentiment Analysis


## 1. Students

- Ayao Anifrani <ayao.anifrani@student.ecp.fr>
- Thomas Fraisse <thomas.fraisse@student.ecp.fr>

## 2. Final System

### 2.1 Required packages
	
In our work, we used, apart from common Python librairies,
-	`spacy`
- 	`pytorch`
-	`transfomers`

### 2.2 Problem modelisation
	
After trying out several approaches, our final method is inspired from one of the approaches in *Sun et al. (2019) [1]*. More precisely, the modelisation that gives us the best accuracy on the dev set is a modified version of the **QA-M** task:
- First we are modelling this Target Aspect-Based Sentiment Analysis classification problem as a sentence pair classification problem
- The sentence pair consists of (it is to be noticed that instead of using the sentences in the order **question-answer** as in the paper, we used **answer-question** for better performance): 
    - the first sentence (**the answer**) is simply the one to be initially classified
    - a sentence (**the question**) formed from the aspect and the category. A question asking the opinion about the given category ; and then the given word of interest.
- For exemple, with the category being `"AMBIENCE#GENERAL"` and the word to be focused on being `"seating"`, the first example in the train set becomes: 
```python
question = "what do you think of the ambience of it ? seating"
answer = "short and sweet – seating is great:it's romantic,cozy and private."
```

The main interest of BERT is that it is very suitable for sentence pair classification. That makes BERT embedding a great idea for us and particularly for the **QA-M** task. We also tested the other task forms but none of them helped improve the classification.

### 2.3 Classifier model

#### a) Text preprocessing

As BERT has been trained to learn a very broad syntax and semantic including from ponctuated sentences, we decided to do only a few preprocessing. Thus, we only removed punctuactions instead of those in `[',', '-', '.', "'", '!']`. For example, deleting the stop words makes the accuracy decreases.

#### b) BERT For Senquence Classification

We fine tuned the `BertForSentenceClassification` pretrained model from the transformers librairy. We used the `base-uncased` embeddings with 12 transfomer blocks with a hidden layer size at 768 (can be considered as embedding size. For example we tried to export this layer's weights and use it as features in different scikit-learn classifiers but this technique poorly performed with 64% of accuracy at best). This model loads 110M parameters then.

We set the initial learning rate at `2e-5` and used a Adam optimizer with weight decay at a factor of 0.01 (not for all weights). We also used a linear scheduler without warm up (`num_warmup_step=0`). The training batch size is 16 and we the training takes 10 epochs.

- Attention mask: Useful for padding. BERT tokenizer sets the mask at 1 for our words and 0 for padded tokens.
- Token type ID: This is useful to make a difference between the two sequences in the pair. Here, the answer's tokens have a ID at 0 and the question's tokens ID are 0.

#### c) Header classifier
		
`BertForSentenceClassification` comes with a linear classifier (a fully conected layer of input size 768). But we used another classifier instead of that one. So the head of the neural network is a Average pooling layer with a kernel size of 24 and a fully connected layer of input size 32).

## 3. Model Performance
	
The accuracy score of the model we finally kept is 77.13 (std=0.75 on 5 tests). Full result: 
```python 
[76.33, 76.86, 78.46, 76.6, 77.39]
```

## 4. References

-	[1] Chi Sun, Luyao Huang, Xipeng Qiu. 2019. *Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence*. NAACL 2019
-	[2] Chi Sun, Xipeng Qiu, Yige Xu, Xuanjing Huang. 2019. *How to Fine-Tune BERT for Text Classification?* arXiv preprint arXiv:1905.05583, 2019
