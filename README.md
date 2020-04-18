# NLP Exercise 2: Aspect-Based Sentiment Analysis


## 1. Students

- Ayao Anifrani <ayao.anifrani@student.ecp.fr>
- Thomas Fraisse <thomas.fraisse@student.ecp.fr>

## 2. Final System

### 2.1 Required packages
	
In our work, we used, apart from native Python librairies,
- 	`pytorch`
-	`transfomers`

### 2.2 Problem modelisation
	
After trying out several approaches, our final method is one of the approaches in *Sun et al. (2019)*. More precisely, the modelisation that gives us the best accuracy on the dev set is the **NLI-M** (as the problem is not exactly the same, we adapt it a little here):
- First we are modelling this Target Aspect-Based classification problem as a sentence pair classification problem
- The sentences are: one sentence (*the question*) formed from the aspect and the category; and the second sentence is simply the initial one
- For exemple, with the category being `"AMBIENCE#GENERAL"` and the word to be focused on being `"seating"`, the first example in the train set becomes (with the separation made with a question mark): 
```python
"seating - ambience ? short and sweet – seating is great:it's romantic,cozy and private."
```

The main interest of BERT is that it is very suitable for sentence pair classification. That makes BERT embedding a great idea for us and particularly for the **NLI-B** task.

### 2.3 Classifier model

#### a) Text preprocessing

#### b) BERT For Senquence Classification

We fine tuned the `BertForSentenceClassification` pretrained model from the transformers librairy. We used the `base-uncased` embeddings with 12 transfomer blocks with a hidden layer size at 768 (can be considered as embedding size. For example we tried to export this layer's weights and use it as features in different scikit-learn classifiers but this technique poorly performed with 64% of accuracy at best). This model loads 110M parameters then.

We set the initial learning rate at `2e-5` and used a Adam optimizer with weight decay at a factor of 0.1 (not for all weights). We also used a linear scheduler without warm up (`num_warmup_step=0`)

- Attention mask: 
- Token type ID: 

#### c) Header classifier
		
Instead of the natural linear classifier used by `BertForSentenceClassification`, we used two fully connected layer to decrease smoothly the neurons from 768 to 64 then from 64 to 3 (the number of labels).

## 3. Model Performance
	
The accuracy score of the model we finally kept is

## 4. References

-	Chi Sun, Luyao Huang, Xipeng Qiu. 2019. *Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence*. NAACL 2019
-	Chi Sun, Xipeng Qiu, Yige Xu, Xuanjing Huang. 2019. *How to Fine-Tune BERT for Text Classification?* arXiv preprint arXiv:1905.05583, 2019
