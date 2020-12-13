# Assessment Modeling

### pre-train code
`main.py`
- mapping 파일 로드 
- train, validation 파일 로드  
- 유저당 샘플 개수 계산(`*_sample_list, num_of_*_user`)
- train, validation 각각 DataSet, DataLoader instance 만들기
- output size 결정 (`network.py`로 옮길 예정)
- model instance 만들기  
- trainer instance 만들기
- 학습 코드 돌리기 (`trainer.train()`)

 
`dataset.py`
- `__getitem__`
  - DataLoader가 data 부를 때 사용하는 method
  - `get_sequence`를 콜
- `get_sequence`
  - data에서 불러야할 start, end index 잡아줌
  - `get_sequence_by_index` 를 통해 데이터를 가져온 후, model length에 맞게 padding 처리
  - `random_assessment` 를 통해, 데이터 중 pre-train task에 맞게,
    - input에 random하게 mask 
    - 학습해야할 index와 label 가져옴
- `get_sequence_by_index`
  - 파일을 열어서, data read
  - input, label에 맞게 index 변환(TRUE_INDEX 등)
- `random_assessment`
  - 데이터 중 pre-train task에 맞게 1) input에 random하게 mask하고 2)학습해야할 index와 label 가져옴

`trainer.py`
- `Trainer` 클래스 선언
  - `train`
    - 정해진 epoch 만큼 model의 `_inference` 를 콜해서 train, validate 수행
    - model weight save
    - wandb logging
  - `_inference`
    - loader를 for loop으로 돌리면서
      - model forward (`time_output`, `start_time_output`, ...)
      - pretrain_task 별로 loss 계산 (구현 예정)
      - batch별 metric 계산
      - `_update` 수행
    - 전체 epoch에 대해 metric 계산
`network.py`
- `Model` (`TwoGeneratorModel`과 병합 예정)
  - network에 필요한 layer들을 init
- `TwoGeneratorModel`
  - Model을 상속받아 forward 진행
  - `forward`
    - input을 `src_embed`에 넣고 콜해서 output 계산

`layers.py`
- 모델에 필요한 layer들 구현

## Downstream codes
   - score_main.py
   - score_dataset.py
   - score_trainer.py
   - score_network.py

## To-dos for LTOP implementation
`main.py`
- num_of_samples 수정해야 할 수 있음(dataset.py의 interaction sample만드는 방법에 따라)
 
`dataset.py`
- interaction sample 만들 때 LT를 고려해 만들어야 함
- 지정한 기준(session, days, weeks)으로 sequence를 LT로 구분
- 두 LT를 가져옴
  - 연속된 것
  - 불연속 된 것
- label 만들기
  - 50%의 확률로 swap
  - swap 안되면 1
  - swap 되면 0
- cls, sep 추가
  - cls, sep 넣었을 때 masking 안되도록 처리 필요
- LT가 (model length - 2)/2 보다 길어질 경우 길이에 맞게 cut(자르는 기준 논의 필요)
- sliding은 추후 논의
 
`model.py`
- LT embedding추가 필요
- LTOP output 계산
 
`trainer.py`
- LTOP에 대한 label받도록 수정
- LTOP output을 통해 loss계산 필요 
- model output나온 이후 loss, update까지의 내용이 pretrain task에 따라 계산되도록 수정

`layers.py`
- EncoderEmbedding 부분의 pretrain task에 해당되는 값들만 계산하도록 수정

`Refactor`
- 필요없는 argument 정리