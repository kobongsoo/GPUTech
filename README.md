# GPUTech
## GPU 메모리보다 큰 모델 파인 튜닝하기
- 모델 훈련 중 "CUDA 메모리 오류.." 문제를 해결하기 위한 메모리를 효율적으로 설정하여 훈련할수 있는 방법에 대해 설명함.
- **GPU 메모리**에 초점을 맞추었음
- 보다 자세한 내용은 [이곳](https://medium.com/@bestasoff/how-to-fine-tune-very-large-model-if-it-doesnt-fit-on-your-gpu-3561e50859af) 참조 하기 바람

### 1. gradient_checkpoint 설정
- backprop(역전파) 기전, 순차적으로 피드포워드 된 노드들의 값을 모두 메모리에 저장해 두는 대신,  체크포인트 방식으로, 필요한 노드들의 값만 메모리에 저장해 두는 방식
- 사용법 : model.**gradient_checkpointing_enable()**
```
gpu_model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
gpu_model.train()
gpu_model.gradient_checkpointing_enable()
    for i in range(epochs):
      훈련시작
```
