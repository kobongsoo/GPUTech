# GPUTech
## PEFT
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

- [Huggingface에서 공개한 PEFT 라이브러리](https://github.com/huggingface/peft)를 이용하면 간단하게 적용할수 있다.
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

## GPU 메모리보다 큰 모델 파인 튜닝하기
- 모델 훈련 중 "CUDA 메모리 오류.." 문제를 해결하기 위한 메모리를 효율적으로 설정하여 훈련할수 있는 방법에 대해 설명함.
- 보다 자세한 내용은 [이곳](https://medium.com/@bestasoff/how-to-fine-tune-very-large-model-if-it-doesnt-fit-on-your-gpu-3561e50859af) 참조 하기 바람
