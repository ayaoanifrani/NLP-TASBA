import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
import spacy
import time
import datetime
import string
import itertools

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nlp = spacy.load("en_core_web_sm")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(768, 3)

    def forward(self, x):
        x = self.fc1(x)
        return x


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class Classifier:
    """The Classifier"""

    #############################################
    def __init__(self, train_batch_size=16, eval_batch_size=8, max_length=128, lr=1e-5, eps=1e-6, n_epochs=11):
        """

        :param train_batch_size:
        :param eval_batch_size:
        :param max_length:
        :param lr:
        :param eps:
        :param n_epochs:
        """
        # model parameters
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.lr = lr
        self.eps = eps
        self.n_epochs = n_epochs

        # Information to be set or updated later
        self.trainset = None
        self.categories = None
        self.labels = None
        self.model = None

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        # The model

        #   We first need to specify some configurations to the model
        configs = BertConfig.from_pretrained('bert-large-uncased', num_labels=3)
        self.model = BertForSequenceClassification(configs)

        #   We are changing the header classifier of BERT (Which is initially a Linear layer)
        #clf = Net()
        #self.model.classifier = clf

        self.model.to(device)  # putting the model on GPU if available otherwise device is CPU

    def preprocess(self, sentence):

        assert isinstance(sentence, str)
        doc = nlp(str(sentence))
        tokens = []
        for token in doc:
            if (not token.is_punct) or (token.text not in [',', '-', '.', "'", '!']):  ## Some punctuations can be interesting for BERT
                tokens.append(token.text)
        tokens = (' '.join(tokens)).lower().replace(" '", "'")

        return tokens

    def question(self, word, category):
        assert category in self.categories

        if category == 'AMBIENCE#GENERAL':
            return "what do you think of " + word.lower() + " and the ambience ? "

        elif category == 'DRINKS#PRICES' or category == 'FOOD#PRICES' or category == 'RESTAURANT#PRICES':
            return "what do you think of " + word.lower() + " and the price ? "

        elif category == 'DRINKS#QUALITY':
            return "what do you think of " + word.lower() + " and the quality of drinks ? "
        elif category == 'DRINKS#STYLE_OPTIONS':
            return "what do you think of " + word.lower() + " and the diversity of drinks ? "
        elif category == 'FOOD#QUALITY':
            return "what do you think of " + word.lower() + " and the quality of food ? "
        elif category == 'FOOD#STYLE_OPTIONS':
            return "what do you think of " + word.lower() + " and the diversity of food ? "
        elif category == 'LOCATION#GENERAL':
            return "what do you think of " + word.lower() + " and the location ? "

        elif category == 'RESTAURANT#GENERAL' or category == 'RESTAURANT#MISCELLANEOUS':
            return "what do you think of " + word.lower() + " and the restaurant ? "

        elif category == 'SERVICE#GENERAL':
            return "what do you think of " + word.lower() + " and the service ? "

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""

        # Loading the data and splitting up its information in lists
        print("Loading training data...")
        trainset = np.genfromtxt(trainfile, delimiter='\t', dtype=str, comments=None)
        self.trainset = trainset
        n = len(trainset)
        targets = trainset[:, 0]
        categories = trainset[:, 1]
        self.labels = list(Counter(targets).keys())  # label names
        self.categories = list(Counter(categories).keys())  # category names
        start_end = [[int(x) for x in w.split(':')] for w in trainset[:, 3]]
        words_of_interest = [trainset[:, 4][i][start_end[i][0]:start_end[i][1]] for i in range(n)]
        sentences = [str(s) for s in trainset[:, 4]]

        # Preprocessing the text data
        print("Preprocessing the text data...")
        sentences = [self.preprocess(sentence) for sentence in tqdm(sentences)]

        # Computing question sequences
        print("Computing questions...")
        questions = [self.question(words_of_interest[i], categories[i]) for i in tqdm(range(n))]

        # Tokenization
        attention_masks = []
        input_ids = []
        token_type_ids = []
        labels = []
        for question, answer in zip(questions, sentences):
            encoded_dict = self.tokenizer.encode_plus(question, answer,
                                                      add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                                      max_length=self.max_length,  # Pad & truncate all sequences
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,  # Construct attention masks
                                                      return_tensors='pt',  # Return pytorch tensors.
                                                      )
            attention_masks.append(encoded_dict['attention_mask'])
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks = torch.cat(attention_masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        # Converting polarities into integers (0: positive, 1: negative, 2: neutral)
        for target in targets:
            if target == 'positive':
                labels.append(0)
            elif target == 'negative':
                labels.append(1)
            elif target == 'neutral':
                labels.append(2)
        labels = torch.tensor(labels)

        # Pytorch data iterators
        train_data = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      batch_size=self.train_batch_size,
                                      sampler=train_sampler)

        # Optimizer and scheduler (we are using a linear scheduler without warm up here)
        no_decay = ['bias', 'gamma', 'beta']  # These parameters are not going to be decreased
        optimizer_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_parameters,
                          lr=self.lr,
                          eps=self.eps)
        total_steps = len(train_dataloader) * self.n_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        # Training
        initial_t0 = time.time()
        for epoch in range(self.n_epochs):
            print('\n======== Epoch %d / %d ========' % (epoch + 1, self.n_epochs))
            print('Training...\n')
            t0 = time.time()
            total_train_loss = 0

            self.model.train()
            for step, batch in enumerate(train_dataloader):

                #if step % 20 == 0 and not step == 0:
                    # Calculate elapsed time.
                    #elapsed = format_time(time.time() - t0)
                    #print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
                batch = tuple(t.to(device) for t in batch)
                input_ids_, input_mask_, segment_ids_, label_ids_ = batch
                self.model.zero_grad()
                loss, _ = self.model(input_ids_,
                                     token_type_ids=segment_ids_,
                                     attention_mask=input_mask_,
                                     labels=label_ids_)
                total_train_loss += loss.item()

                loss.backward()
                # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            training_time = format_time(time.time() - t0)
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epoch duration: {:}".format(training_time))
        print("  Total training time: {:}".format(format_time(time.time() - initial_t0)))

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """

        # Loading the data and splitting up its information in lists
        evalset = np.genfromtxt(datafile, delimiter='\t', dtype=str, comments=None)
        m = len(evalset)
        categories = evalset[:, 1]
        start_end = [[int(x) for x in w.split(':')] for w in evalset[:, 3]]
        words_of_interest = [evalset[:, 4][i][start_end[i][0]:start_end[i][1]] for i in range(m)]
        sentences = [str(s) for s in evalset[:, 4]]

        # Preprocessing the text data
        print("Preprocessing the text data...")
        sentences = [self.preprocess(sentence) for sentence in tqdm(sentences)]

        # Computing question sequences
        print("Computing questions...")
        questions = [self.question(words_of_interest[i], categories[i]) for i in tqdm(range(m))]

        # Tokenization
        attention_masks = []
        input_ids = []
        token_type_ids = []
        for question, answer in zip(questions, sentences):
            encoded_dict = self.tokenizer.encode_plus(question, answer,
                                                      add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                                      max_length=self.max_length,  # Pad & truncate all sequences
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,  # Construct attention masks
                                                      return_tensors='pt',  # Return pytorch tensors.
                                                      )
            attention_masks.append(encoded_dict['attention_mask'])
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict['token_type_ids'])
        attention_masks = torch.cat(attention_masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)

        # Pytorch data iterators
        eval_data = TensorDataset(input_ids, attention_masks, token_type_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data,
                                     batch_size=self.eval_batch_size,
                                     sampler=eval_sampler)

        # Prediction
        named_labels = []
        self.model.eval()
        for batch in eval_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch

            with torch.no_grad():
                logits = self.model(input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask)[0]

            logits = softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)

            # converting integer labels into named labels
            for label in outputs:
                if label == 0:
                    named_labels.append('positive')
                elif label == 1:
                    named_labels.append('negative')
                elif label == 2:
                    named_labels.append('neutral')

        return np.array(named_labels)
