
import dynet as dy
import numpy as np

import torch
from transformers import AlbertModel, AlbertTokenizer

class Albert():

    def __init__(self, pretrained_model='albert-base-v2'):
        """
        :parameter pretrained_model: available models:
           'albert-base-v1'
           'albert-large-v1'
           'albert-xlarge-v1'
           'albert-xxlarge-v1'
           'albert-base-v2'
           'albert-large-v2'
           'albert-xlarge-v2'
           'albert-xxlarge-v2'
        """
        print("Loading Albert model {}".format(pretrained_model))
        self.tokenizer = AlbertTokenizer.from_pretrained(pretrained_model)
        # Models can return full list of hidden-states & attentions weights at each layer
        self.model = AlbertModel.from_pretrained(pretrained_model,
                                                 output_hidden_states=True)
                                                 #output_attentions=True)

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    @property
    def emb_dim(self):
        return self.model.config.hidden_size

    # def get_sentence_embeddings(self, sentence):
        # # example wordpieces
        # # "Ruy writes his book" --> ['▁ru', 'y', '▁writes', '▁his', '▁book']
        
        # # do not add [CLS] and [SEP]
        # sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
        # sentence_wordpieces = self.tokenizer.convert_ids_to_tokens(sentence_ids)
        # sentence_ids_tsr = torch.tensor([sentence_ids])

        # if torch.cuda.is_available():
        #     sentence_ids_tsr = sentence_ids_tsr.to('cuda')
        # with torch.no_grad():
        #     last_hidden_state, pooler_output, all_hidden_states = self.model(sentence_ids_tsr)
        # # Models outputs are now tuples
        # # >>> len(all_hidden_states)
        # # 13
        # # >>> all_hidden_states[0].shape
        # # torch.Size([1, 6, 768])
        # # >>> len(sentence)
        # # 6

        # # reassemble the wordpieces
        # sentence_embeddings = []

        # for i, wordpiece in enumerate(sentence_wordpieces):
        #     # average on all layers
        #     # embedding = torch.mean(torch.stack([all_hidden_states[index_layer][0][i] for index_layer in range(len(all_hidden_states))]), 0)
        #     embedding = torch.stack([layer[0,i] for layer in all_hidden_states]).mean(0)
        #     if wordpiece.startswith('▁'): # new token (not '_')
        #         sentence_embeddings.append(embedding)
        #     else:
        #         sentence_embeddings[-1] += embedding
        # return sentence_embeddings


    def get_sentence_representation(self, sentence):
        """
        Looks up the sentence representation for the given sentence.
        :param sentence: String of space separated tokens.
        :return: Albert.Sentence object
        """
        # example wordpieces
        # "Ruy writes his book" --> ['[CLS]', '▁ru', 'y', '▁writes', '▁his', '▁book', '[SEP]']]
        
        # adding [CLS] and [SEP] helps
        sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
        # use max_length to avoid error:
        # tokenization_utils.py:1329] Token indices sequence length is longer than the specified maximum sequence length for this model (551 > 512)
        max_position_embeddings = self.model.config.max_position_embeddings
        if len(sentence_ids) > max_position_embeddings:
            raise ValueError("sentence longer than %d" % max_position_embeddings)
        sentence_wordpieces = self.tokenizer.convert_ids_to_tokens(sentence_ids)
        sentence_ids_tsr = torch.tensor([sentence_ids])

        with torch.no_grad():
            last_hidden_state, pooler_output, all_hidden_states = self.model(sentence_ids_tsr)
        # Models outputs are now tuples
        # >>> len(all_hidden_states)
        # 13
        # >>> all_hidden_states[0].shape
        # torch.Size([1, 7, 768])
        # >>> len(sentence)
        # 7

        # reassemble the wordpieces
        sentence_embeddings = []

        for i in range(1, len(sentence_wordpieces)-1): # skip CLS, SEP
            wordpiece = sentence_wordpieces[i]
            # average on all layers
            # embedding = torch.mean(torch.stack([all_hidden_states[index_layer][0][i] for index_layer in range(len(all_hidden_states))]), 0)
            embedding = torch.stack([layer[0,i] for layer in all_hidden_states]).mean(0)
            if wordpiece.startswith('▁'): # new token (not '_')
                sentence_embeddings.append(embedding)
            else:
                sentence_embeddings[-1] += embedding

        return Albert.Sentence(sentence_embeddings)


    class Sentence():

        def __init__(self, embeddings):
            """
            :param embeddings: list of sent_len [hidden_size]
            """
            self.embeddings = embeddings

        def __getitem__(self, i):
            """
            Return the embedding for the word :param i:.
            :param i: Index of word in the sentence.
            :return: Embedding for the word.
            """

            device = 'cuda' if torch.cuda.is_available() else ''
            return dy.inputTensor([self.embeddings[i].cpu().numpy()], device)

