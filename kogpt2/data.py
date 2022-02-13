import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class ReadDataset(Dataset):
	"""web novel dataset"""

	def __init__(self, file_path, tokenizer):
		self.file_path = file_path
		self.data = []
		self.tokenizer = tokenizer
		file = open(self.file_path, 'r', encoding='utf-8')

		with open(file_path, "r", encoding="UTF-8") as file:
			datasets = file.read().splitlines()
		print("checkpoint) Data file loaded")
			
		print("Run sentencepiece...")
		print_first = True
		bos_token = tokenizer.get_vocab()[tokenizer.bos_token]
		question_token = tokenizer.get_vocab()["<unused0>"]
		answer_token = tokenizer.get_vocab()["<unused1>"]
		eos_token = tokenizer.get_vocab()[tokenizer.eos_token]
		for line in tqdm(datasets):
			if print_first:
				tqdm.write("First line of data:")
				tqdm.write(str(line))
			line = line.split("\t")
			tokenized_lines = tokenizer(line)["input_ids"]
			# BOS
			index_of_words = [bos_token]
			# Context
			index_of_words += tokenized_lines[0]
			if len(tokenized_lines) % 2 != 1:
				tqdm.write("Error: question and answer pair is not properly matched")
				tqdm.write(str(line))
				continue
			for i in range(1, len(line), 2):
				# Question...
				index_of_words += [question_token]
				index_of_words += tokenized_lines[i]
				# And its answer
				index_of_words += [answer_token]
				index_of_words += tokenized_lines[i+1]
			index_of_words += [eos_token]
			if len(index_of_words) > 1000:
				continue
			if print_first:
				tqdm.write("...is tokenized to:")
				tqdm.write(str(index_of_words))
			index_of_words = torch.tensor(index_of_words, dtype=torch.long)
			self.data.append(index_of_words)
			print_first = False
		print("Finished!")

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		item = self.data[index]
		return item