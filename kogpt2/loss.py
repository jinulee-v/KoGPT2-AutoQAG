import torch
import torch.nn.functional as F

# Index of KoGPT2 Tokenizer special tokens
SEP_TOKEN=1   # </s>
QUES_TOKEN=9  # <unused0>
ANS_TOKEN=10  # <unused1>

def parse_index(inputs):
  """
  Returns each question and answer tensors
  """
  questions, answers = [], []
  for i, tok in enumerate(inputs):
    if i == 0:
      continue
    if tok == QUES_TOKEN or tok == SEP_TOKEN:
      questions.append(i)
    elif tok == ANS_TOKEN:
      answers.append(i)
  assert len(answers) >= 1
  assert len(questions) == len(answers) + 1

  return questions, answers
        #  body:      logits[:questions[0]]
        #  questions: torch.cat([logits[q+1:a] for q, a in zip(questions, answers)], dim=0)
        #  answers:   torch.cat([logits[a+1:q] for q, a in zip(answers, questions[1:])], dim=0)
        #  <unused0>: torch.cat([logits[q] for q in questions], dim=0)
        #  <unused1>: torch.cat([logits[a] for a in answers], dim=0)

def question_gen_loss(inputs, logits, questions, answers):
  """
  Only retrieve QAG part generation loss, excluding the context
  """
  start_index = questions[0] + 1
  end_index = answers[-1]
  # shift inputs
  inputs = inputs[start_index+1:end_index+1]
  logits = logits[start_index:end_index]
  return F.cross_entropy(logits, inputs)
  

def answer_in_context_loss(inputs, logits, questions, answers):
  """
  Penalize gpt model when it does not generate answer from the context.
  Instead of strict copy generation, we softly train the model to find the answer from the context.
  """
  body_tokens = list(set(inputs[1:questions[0]])) + [QUES_TOKEN, SEP_TOKEN]

  # Set mask
  mask = torch.ones(logits.size(1)).to(logits.device)
  mask[body_tokens] = 0
  mask.requires_grad = False

  answer_logits = torch.cat([logits[a+1:q] for q, a in zip(answers, questions[1:])], dim=0)
  answer_logits = F.softmax(answer_logits, dim=1)

  loss = torch.sum(answer_logits * mask.unsqueeze(0))

  return loss


def sep_token_loss(inputs, logits, questions, answers):
  """
  Penalize GPT model when it does not generate <unused0> and <unused1> tokens in correct order.
  """

def question_diversity(inputs, logits, questions, answers):
  """
  Penalize GPT model when it generates similar questions.
  """


def lossQAG(input_batch, logit_batch, loss_dict):

  loss = 0
  for i in range(input_batch.size(0)):
    inputs = input_batch[i]
    logits = logit_batch[i]

    questions, answers = parse_index(input_batch[i])

    for loss_fn in loss_dict:
      loss += globals()[loss_fn](inputs, logits, questions, answers) * loss_dict[loss_fn] # FIXME: globals() may not be secure method to call by name
  return loss