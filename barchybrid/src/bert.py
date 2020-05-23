
import dynet as dy
import numpy as np

import torch
from transformers import BertModel, BertTokenizer


def entropy(p):
    """ Compute the entropy of a probability distribution """
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


class Bert():

    def __init__(self, pretrained_model, config=None, pretrained_tokenizer='bert-base-multilingual-cased'):
        print("Loading Bert model {}".format(pretrained_model))
        uncased = 'uncased' in pretrained_tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer, do_lower_case=uncased)
        # Models can return full list of hidden-states & attentions weights at each layer
        self.model = BertModel.from_pretrained(pretrained_model, config=config,
                                               output_hidden_states=True,
                                               output_attentions=True)

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    @property
    def emb_dim(self):
        return self.model.config.hidden_size

    # def get_sentence_embeddings(self, sentence):
    #     """ :param sentence: a string. """
    #     # example wordpieces
    #     # "Ruy writes his book" --> ['[CLS]', 'ru', '##y', 'writes', 'his', 'book', '[SEP]']
        
    #     # adding [CLS] and [SEP] helps
    #     sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
    #     sentence_wordpieces = self.tokenizer.convert_ids_to_tokens(sentence_ids)
    #     sentence_ids_tsr = torch.tensor([sentence_ids])
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     sentence_ids_tsr = sentence_ids_tsr.to(device)

    #     with torch.no_grad():
    #         loss, logits, all_hidden_states, all_attentions = self.model(sentence_ids_tsr)
    #     # Models outputs are now tuples
    #     # >>> len(all_hidden_states)
    #     # 12
    #     # >>> all_hidden_states[0].shape
    #     # torch.Size([1, 7, 768])
    #     # >>> len(sentence_ids)
    #     # 7

    #     # reassemble the wordpieces
    #     sentence_embeddings = []
    #     for i in range(1, len(sentence_wordpieces)-1): # skip CLS, SEP
    #         wordpiece = sentence_wordpieces[i]
    #         # average on all layers
    #         embedding = torch.stack([layer[0,i] for layer in all_hidden_states]).mean(0)
    #         if wordpiece.startswith('##'): # compound token
    #             sentence_embeddings[-1] += embedding
    #         else:
    #             sentence_embeddings.append(embedding)

    #     # attention
    #     # all_attention = [ [batch, sentlen, sentlen] ]
    #     # drop the batch dimension and CLS,SEP
    #     attns = [attn[0][1:-1,1:-1] for attn in all_attentions]

    #     return sentence_embeddings, attns


    def get_sentence_representation(self, sentence):
        """
        Looks up the sentence representation for the given sentence.
        :param sentence: String of space separated tokens.
        :return: Bert.Sentence object
        """
        # example wordpieces
        # "Ruy writes his book" --> ['[CLS]', 'ru', '##y', 'writes', 'his', 'book', '[SEP]']
        
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
            loss, logits, all_hidden_states, all_attentions = self.model(sentence_ids_tsr)
        # Models outputs are now tuples
        # >>> len(all_hidden_states)
        # 12
        # >>> all_hidden_states[0].shape
        # torch.Size([1, 7, 768])
        # >>> len(sentence_ids)
        # 7

        # all_hidden_states (a list of layers) = [ [batch, sent_len, emb_dim] ]
        # reassemble the wordpieces
        sentence_embeddings = []
        for i in range(1, len(sentence_wordpieces)-1): # skip CLS, SEP
            wordpiece = sentence_wordpieces[i]
            # average on all layers
            embedding = torch.stack([layer[0,i] for layer in all_hidden_states]).mean(0)
            if wordpiece.startswith('##'): # compound token
                sentence_embeddings[-1] += embedding
            else:
                sentence_embeddings.append(embedding)

        # attention
        # all_attention (a list of layers) = [ [batch, num_heads, sent_len, sent_len] ]
        # drop the batch dimension and CLS,SEP row/col.
        attns = torch.stack([ attn[0,:,1:-1,1:-1] for attn in all_attentions ])
        # normalize rows
        attns = attns/attns.sum(axis=3, keepdims=True)
        return Bert.Sentence(sentence_embeddings, attns)
    

    class Sentence():

        def __init__(self, embeddings, attentions):
            """
            :param embeddings: list of sent_len [hidden_size]
            :param attentions: list of layers [num_heads, sent_len, sent_len]
            """
            self.embeddings = embeddings
            self.attentions = attentions


        def __getitem__(self, i):
            """
            Return the embedding for the word :param i:.
            :param i: Index of word in the sentence.
            :return: Embedding for the word.
            """

            device = 'cuda' if torch.cuda.is_available() else ''
            return dy.inputTensor([self.embeddings[i].cpu().numpy()], device) # slow down!

