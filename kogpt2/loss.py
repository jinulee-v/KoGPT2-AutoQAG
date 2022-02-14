import torch

def parse_logit(input, logit):
  

def answer_in_context_loss(inputs, logits):
  """
  Penalize gpt model when it does not generate answer from the context.
  """

def sep_token_loss(inputs, logits):
  """
  Penalize GPT model when it does not generate <unused0> and <unused1> tokens in correct order.
  """

def question_diversity(inputs, logits):
  """
  Penalize GPT model when it generates similar questions.
  """