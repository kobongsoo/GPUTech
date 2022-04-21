# GPUTech
## GPU 메모리보다 큰 모델 파인 튜닝하기
- 모델 훈련 중 "CUDA 메모리 오류.." 문제를 해결하기 위한 메모리를 효율적으로 설정하여 훈련할수 있는 방법에 대해 설명함.
- **GPU 메모리**에 초점을 맞추었음
- 보다 자세한 내용은 [이곳](https://medium.com/@bestasoff/how-to-fine-tune-very-large-model-if-it-doesnt-fit-on-your-gpu-3561e50859af) 참조 하기 바람

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
### 4. Mixed-precision training(혼합 정밀도 훈련)
- 훈련은 gpu 모델을 이용하는 데신, **GPU 모델 크기를 반으로 줄여 사용**(half())하고, 대신 **optimizer는 CPU 모델을 이용** 하는 방식
-  **optimizer는 훈련된 GPU 모델의 grad를 CPU모델로 복사 후, 감소 시킴**
-  **CPU 모델 state_dict을 GPU 모델로 state_dict로 로딩**하는 과정 필요
- CPU 모델과 GPU 모델 2개 필요. **훈련시간이 GPU 일때 보다 엄청 증가함(8~10배)**

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
