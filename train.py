import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModelForCausalLM
from kogpt2.sample import sample_sequence
from kogpt2.data import ReadDataset
from kogpt2.loss import lossQAG
from tqdm import tqdm
import subprocess
import os
import re
import argparse
import numpy as np
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200,
					help="epoch 를 통해서 학습 범위를 조절합니다.")
parser.add_argument('--save_path', type=str, default='./checkpoint/',
					help="학습 결과를 저장하는 경로입니다.")
parser.add_argument('--load_path', type=str, default=None, #
					help="학습된 결과를 불러오는 경로입니다.")
parser.add_argument('--samples', type=str, default="samples/",
					help="생성 결과를 저장할 경로입니다.")
parser.add_argument('--data_file_path', type=str, default='./raw_data/train.txt',
					help="학습할 데이터를 불러오는 경로입니다.")
parser.add_argument('--batch_size', type=int, default=1,
					help="batch_size 를 지정합니다.")
parser.add_argument('--update_freq', type=int, default=128,
					help="update_freq 를 지정합니다.")
args = parser.parse_args()



def get_gpu_memory_map():
	"""Get the current gpu usage.

	Returns
	-------
	usage: dict
		Keys are device ids as integers.
		Values are memory usage as integers in MB.
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		], encoding='utf-8')
	# Convert lines into a dictionary
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map

def main(epoch, save_path, load_path, samples, data_file_path, batch_size, update_freq):
	ctx = 'cuda'
	cachedir = 'cache/'

	print("\n============================\n")
	print("KoGPT-2 Transfer Learning Initialization process...")
	print("\n============================\n")

	tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2", cache_dir=cachedir, \
	                                          bos_token='</s>', eos_token='</s>', unk_token='<unk>', \
			                                      pad_token='<pad>', mask_token='<mask>')

	count = 0
	if load_path is None:
		model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2", cache_dir=cachedir)
	else:
		# 불러오기 부분
		try:
			model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2", cache_dir=cachedir)
			model.load_state_dict(torch.load(load_path)["model_state_dict"])
			count = int(re.findall("\d+", load_path)[1])
		except:
			exit()
	print("Model size configuration:")
	print(model)

	device = torch.device(ctx)
	model.to(device).train()


	print("Current updates: ", count)

	dataset = ReadDataset(data_file_path, tokenizer)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=lambda x: pad_sequence(x, batch_first=True))
	print("checkpoint) DataLoader Prepared")



	learning_rate = 3e-3
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	optimizer.zero_grad()

	print("\n============================\n")
	print("KoGPT-2 Transfer Learning Start")
	print("\n============================\n")

	update_count = 0
	recent_loss = []
	recent_loss_cnt = 100
	for ep in range(epoch):
		print("Epoch: ", ep)
		for data in tqdm(data_loader):
			
			data = data.to(device)

			try:
				outputs = model(data, labels=data)
				logits = outputs.logits
				loss = lossQAG(data, logits, ["question_gen_loss"])
				loss.backward()
			except RuntimeError as e:
					if 'out of memory' in str(e):
							tqdm.write('WARNING: ran out of memory, retrying batch')
							del outputs
							del loss
							for p in model.parameters():
									if p.grad is not None:
											del p.grad  # free some memory
							gc.collect()
							torch.cuda.empty_cache()
							# retry
							outputs = model(data, labels=data)
							loss, logits = outputs[:2]
							loss = loss.to(device)
							loss.backward()
					else:
							raise e
			recent_loss.insert(0, float(loss))
			if len(recent_loss) > recent_loss_cnt:
				recent_loss = recent_loss[:recent_loss_cnt]

			# delayed update
			update_count += 1
			if update_count == update_freq:
				optimizer.step()
				update_count = 0
				count += 1
				optimizer.zero_grad()
			
			# logging
			if update_count == 0 and count % 10 == 0 and count > 0:
				tqdm.write('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(ep, count, loss, sum(recent_loss) / len(recent_loss)))

			# generator test
			if (update_count == 0 and count > 0 and count % 100 == 0) or (len(data) < batch_size):
				test_sent = "행정안전부장관은 매년 18세가 되는 남성에 대하여 병역준비역 편입자의 조사에 필요한 주민등록 정보화자료를 병무청장에게 통보하여야 한다. 병무청장은 주민등록이 되어 있지 아니한 사람 등에 대한 병역준비역 편입자 조사를 위하여 법원행정처장에게 매년 18세가 되는 남성의 가족관계등록 정보화자료를 요청할 수 있다. 제1항에 따른 정보화자료 통보의 범위 및 절차 등에 필요한 사항과 병역준비역 편입자로서 국외출생(國外出生) 등의 사유로 주민등록이 되어 있지 아니한 사람의 조사 등에 필요한 사항은 대통령령으로 정한다. 제1항에 따른 병역준비역 편입자의 조사에 필요한 사항은 병무청장이 정한다."
				sent = sample_sequence(model, tokenizer, tokenizer.get_vocab(), sent=test_sent, text_size=100, temperature=0.7, top_p=0.8, top_k=40)
				tqdm.write(sent)

			#########################################

			if (count > 0 and count % 1000 == 0) or (len(data) < batch_size):
				# save model
				try:
					torch.save({
						'epoch': epoch,
						'train_no': count,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'loss': loss
					}, save_path + 'KoGPT2_checkpoint_' + str(count) + '.pt')
				except:
					pass

if __name__ == "__main__":
	main(args.epoch, args.save_path, args.load_path, args.samples, args.data_file_path, args.batch_size, args.update_freq)
