# KoGPT2-AutoQAG(한국어 질의응답 자동생성)
### 해군미래혁신연구단 '22. 

-----------
### description
  - SKT-AI에서 한국어 데이터를 Pre-Training 시킨 KoGPT2를 fine tuning하여 Context 기반 질의응답 자동 생성
  - 개인/소규모 개발팀 등 데이터셋 확보 어려운 조직에서 QA 챗봇 시스템 구축하기 위해 사용 가능

----------

### input data structure
  ```</s>컨텍스트<unused0>질문1<unused1>답변1<unused0>질문2<unused1>답변2 ... </s>```
  
---------

### how to install
  ```sh
git clone https://github.com/jinulee-v/KoGPT2-AutoQAG
cd KoGPT2-finetuning
pip install -r requirements.txt
  ```

----------

### fine tuning
  - 데이터 다운로드 <br>
    SQuAD json형식을 따르는 데이터들을 raw_data 폴더 내에 추가. (예시로 KorQuAD 1.0 dev set 추가됨) <br>
    KorQuAD 1.0: https://korquad.github.io/KorQuad%201.0/ <br>
    AI Hub 도서자료 기계독해: https://aihub.or.kr/aidata/30715 <br>
    `python generate_tsv.py`
  - KoGPT2 fine tuning
    `python main.py --data_file_path=raw_data/train.txt --save_path=./checkpoint/` <br>
    `python main.py --data_file_path=raw_data/train.txt --save_path=./checkpoint/ --load_path=./checkpoint/KoGPT2_checkpoint_80000.pt` <br>
    *load_path는 존재하면 학습된 모델부터 train 시작 / 존재하지않으면 처음부터 train 시작*
  
----------
### fine tuning - output example
  TO BE UPDATED

----

### generator
  TO BE UPDATED

----------

### reference 
  - [KoGPT2](https://github.com/SKT-AI/KoGPT2)
