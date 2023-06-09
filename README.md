<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/><img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
## < PEFT >
- Parameter Efficient Fine-Tuning(PEFT): 모델의 모든 파라미터를 튜닝하는 것이 아닌 일부 파라미터만을 튜닝함으로써 모델의 성능을 적은 자원으로도 높게 유지하는 방법론
- PEFT 기법으로 LoRA, IA3, prompt tuning, prefix tuning 등이 있음.
### LoRA
- LoRA(Low-Rank Adaptation)의 개념을 간단하게 설명하자면, 고정된 weights를 갖는 pretrained model에 학습이 가능한 rank decomposition 행렬을 삽입한것으로
중간중간 학습이 가능한 파라미터를 삽입했다는 점에서는 어댑터와 비슷하지만 구조적으로 조금 다르다고 할 수 있습니다.
- 적은 양의 파라미터로 모델을 튜닝하는 방법론이기 때문에 적은수의 GPU로 빠르게 튜닝할 수 있다는 장점이 있습니다.
- LoRA에서 나온 rank decomposition이라는 말이 처음에는 어렵게 느껴졌었는데요.
아래 보이는 그림에서 처럼 행렬의 차원을 r 만큼 줄이는 행렬과 다시 원래 크기로 키워주는 행렬의 곱으로 나타내는 것을 의미합니다.

![image](https://github.com/kobongsoo/GPUTech/assets/93692701/0dfd007a-01c6-4922-b90d-c8a617015fb7)

참고: https://devocean.sk.com/blog/techBoardDetail.do?ID=164779&boardType=techBlog

### LoRA by Huggingface
- [Huggingface에서 공개한 PEFT 라이브러리](https://github.com/huggingface/peft)를 이용하면 LoRA 기법을 간단하게 적용할수 있다.
```
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```
- Huggingface 지원 TaskType과 AutoModelForxxx 함수 
```
class TaskType(str, enum.Enum):
 	SEQ_CLS = "SEQ_CLS"
    	SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    	CAUSAL_LM = "CAUSAL_LM"
    	TOKEN_CLS = "TOKEN_CLS"

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
```
예제 : **[GPT-LoRA (LLM 모델을 파인튜닝하는 예시)](https://github.com/kobongsoo/GPUTech/blob/master/GPT-LoRA.ipynb)**
<br>참고 소스
<br> https://github.com/quantumaikr/KoreanLM
<br> https://github.com/jeremyarancio/llm-rpg/

## <GPU 메모리보다 큰 모델 파인 튜닝하기>
- 모델 훈련 중 "CUDA 메모리 오류.." 문제를 해결하기 위한 GPU 메모리를 효율적으로 설정하여 훈련할수 있는 방법에 대해 설명함.
- **GPU 메모리**에 초점을 맞추었음
- 보다 자세한 내용은 [이곳 사이트](https://medium.com/@bestasoff/how-to-fine-tune-very-large-model-if-it-doesnt-fit-on-your-gpu-3561e50859af) 혹은 [파일](https://github.com/kobongsoo/GPUTech/blob/master/reference/GPU%EC%97%90%20%EB%A7%9E%EC%A7%80%20%EC%95%8A%EB%8A%94%20%EB%A7%A4%EC%9A%B0%20%ED%81%B0%20%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EB%AF%B8%EC%84%B8%20%EC%A1%B0%EC%A0%95%ED%95%98%EB%8A%94%20%EB%B0%A9%EB%B2%95.pdf)참조 하기 바람
- 아래 표는 실제, [BERT 테스트 소스](https://github.com/kobongsoo/GPUTech/blob/master/bert-fpt-gpu-test.ipynb)를 직접 구현해서 테스트 해본 결과임

##### [ hyper-Parameter : batch_size=32, lr=3e-5, epochs=3, 모델: BertMLMModel, 서버: 24G GPU 서버, traindata: 10,000 문장/evaldata: 110문장]
|방식|GPU사용량(기준:22G)|훈련속도/1epoch(기준: 110초)|평가 정확도(기준:89%)|
|:------|:---:|:---:|:---:|
|1.gradient_checkpoint|21.3G|110|89%|
|**2.micro_batch/Gradient accumulation**|9.5G|180|89%|
|3.8bit-adam optimizer|21.8G|100|89%|
|4.Mixed-precision training|12G|420|90%|
|1+2+3|8G|185|89%|
|1+2+4|4.6G|400|90%|

- **2.micro_batch/Gradient accumulation 는 batch_size=8 로 하고, accumulation_steps = 4로 한 경우임**
- **3.8bit-adam optimizer와  4.Mixed-precision training 같이 적용 하면 안됨** 
   - 같이 적용시 에러 발생함-'Error an illegal memory access was encountered at line 276 in file /private/home/timdettmers/git/bitsandbytes/csrc/ops.cu' (**원인 모름**)		
				
				
				
### 1. gradient_checkpoint 설정
- backprop(역전파) 기전, 순차적으로 피드포워드 된 노드들의 값을 모두 메모리에 저장해 두는 대신,  체크포인트 방식으로, 필요한 노드들의 값만 메모리에 저장해 두는 방식
- 사용법 : model.**gradient_checkpointing_enable()**
```
model = BertForMaskedLM.from_pretrained(model_path)
model.train()
model.gradient_checkpointing_enable()
    for i in range(epochs):
      훈련시작
```
참고 소스 : [groadient_checkpointing.py](https://github.com/kobongsoo/GPUTech/blob/master/reference/gradient_checkpointing.py)

### 2. micro_batch/Gradient accumulation(마이크로 배치&기울기 축적)
- 배치사이즈는 8로 하고, 대신에 accumulation_steps = 4로 해서, 4번 기울기 축적을 누적해서 32배치사이즈와 동일한 효과를 주는 방식
- 1훈련 과정에서 손실 계산은 **loss = loss/accumulation_steps** 이 됨
- accumulation_steps 누적해서 한꺼번에 optimizer step 시킴
- **훈련시간이 1.4배** 더 증가함
```
batch_size = 8
accumulation_steps = 4

model = BertForMaskedLM.from_pretrained(model_path)
model.train()

  for i in range(epochs):
    for batch_idx, data in enumerate(tqdm(train_loader)):
       # 모델 실행
        outputs = model(input_ids=input_ids, 
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids,
                       labels=labels)
        
        # 출력값 loss를 outputs에서 얻어옴
        loss = outputs.loss
        
        # ** 손실값을 accumulation_steps 나눔
        loss = loss / accumulation_steps  
        
        loss.backward()   # backward 구함
        
        # optimizer는 accumulation_steps 주기로, 업데이트 시킴
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
           optimizer.step()  # 가중치 파라미터 업데이트(optimizer 이동)
           scheduler.step()  # 학습률 감소
           model.zero_grad()# 그래디언트 초기화
                 
```

참고 소스 : [gradient_accum_training_loop.py](https://github.com/kobongsoo/GPUTech/blob/master/reference/gradient_accum_training_loop.py)

### 3. 8bit-adam optimizer 사용
- 기존 32bit Adam 옵티마이져 대신 8bit Aadmin 옵티마이져를 사용하여, 2^32 -> 2^8 으로 메모리를낮춤
- **bitsandbytes 라이브러리 설치** 해야함
- **bert 모델에서 훈련 테스트시 gpu 메모리가 낮아지지는 않았음**
- 사용방법 : optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.995))
```
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.995))
```

```
import bitsandbytes as bnb
adam = bnb.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995), optim_bits=8) 
```
- bitsandbytes 라이브러리 설치
  - 8-bit optimizer 사용을 위해 cuda 버전을 확인하고, 해당 버전에 맞는 bitsandbytes 라이브러리 설치함
  - Bitsandbytes is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers and quantization functions.
  - 참조 : [bitsandbytes](https://github.com/facebookresearch/bitsandbytes)
```
!conda list | grep cudatoolkit     # cuda 버전 확인
!pip install bitsandbytes-cuda113  # 해당 버전에 맞는 bitsandbytes 설치 (버전이 11.3 이면, cuda113 으로 설치)
```
참고 소스 : [8-bit-adam](https://github.com/kobongsoo/GPUTech/blob/master/reference/8-bit-adam.py)

### 4. Mixed-precision training(혼합 정밀도 훈련)
- 훈련은 gpu 모델을 이용하는 데신, **GPU 모델 크기를 반으로 줄여 사용**(half())하고, 대신 **optimizer는 CPU 모델을 이용** 하는 방식
-  **optimizer는 훈련된 GPU 모델의 grad를 CPU모델로 복사 후, 감소 시킴**
-  **CPU 모델 state_dict을 GPU 모델로 state_dict로 로딩**하는 과정 필요
- CPU 모델과 GPU 모델 2개 필요. **훈련시간이 GPU 일때 보다 5배 정도 증가함**

```
# GPU 모델 생성
model = BertForMaskedLM.from_pretrained(model_path).half()  # gpu 모델은 반으로 줄임(half())
model.to('CUAD:0')

# CPU 모델 생성 
cpu_model = BertForMaskedLM.from_pretrained(model_path)     
model.to('cpu')

# Optimizer는 CPU 모델로 지정
optimizer = AdamW(cpu_model.parameters(), lr=3e-5,  eps=1e-8)

model.train()

for i in range(epochs):
    for batch_idx, data in enumerate(tqdm(train_loader)):
    
       # GPU 모델 훈련
       outputs = model(**input)
       loss = outputs.loss
       loss.backward()   # backward 구함
       
       # ** optimizer는 훈련된 GPU 모델의 grad를 CPU모델로 복사후, 감소 시킴
       # 훈련된 gpu 모델의 grad를 cpu 모델의 grad로 복사
       for p_gpu, p_cpu in zip(model.parameters(), cpu_model.parameters()):
           p_cpu.grad = p_gpu.grad.cpu().to(torch.float32)
        
       #  optimzer 감소  
       model.zero_grad()
       optimizer.step()
       scheduler.step()  
       
       # CPU 모델 state_dict을 GPU 모델로 로딩함
       model.load_state_dict(cpu_model.state_dict()) 
       cpu_model.zero_grad()
```
참고 소스 : [mixed-precision.py](https://github.com/kobongsoo/GPUTech/blob/master/reference/mixed-precision.py)

***
## <[Colossal-AI](https://colossalai.org/)>
- AI 모델을 훈련하기 위한 다양한 병렬 처리 방법을 제공함으로써, 저사양 H/W에서 대규모 모델 훈련 및 배포 할수 있도록 하는 Open 소스
- 기존 솔루션과 비교하여 GPU 효율을 10배이상 늘릴 수 있다고 함
 <br> 예로 GPU 1개로 180억 매개변수(GPT-2-EXTRALARGE 모델 10배 큰 모델) AI 모델 훈련 할수 있다고 함
