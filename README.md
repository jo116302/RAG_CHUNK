# RAG CHUNK
## Chunk 옵션
- `sentences: List[str]`: Chunk 진행할 문장 단위의 리스트
- `sim_threshold: Optional[float]`: 단일 Chunk 작성을 위한 유사도 -1 부터 1 사이 값을 가지며, 1에 가까울 수록 유사도가 높음
- `overlap_threshold: Optional[float]`: Chunk 간 Overlap(중복) 구간을 구성하기 위한 유사도 범위   
- `check_subtitle_maxlength: Optional[int]`: `<h[n]>` 태그로 시작하는 문장은 소제목으로 간주하여 check_subtitle_maxlength 보다 작으면 이전 문장과 동인 Chunk로 결합하며, 큰 경우 신규 Chunk로 구성
- `min_chunk_tokens: Optional[int]`: 단일 Chunk의 최소한의 token 크기로 너무 작은 청크로 분리되는 것을 방지하기 위한 값
- `max_chunk_tokens: Optional[int]`: 단일 Chunk가 구성될 수 있는 최대 크기의 token 값
- `compare_sentence_size: Optional[int]`: compare_sentence_size 이상의 이전 문장과 현재 문장의 유사도를 비교 (compare_sentence_size개의 개별 문장과 현재 문장의 평균 유사도 비교) 

## 파일 리스트
- `main.py`: url의 정책을 정의하며, 맵핑을 정의
- `Chunking.py`: Chunk 구현한 비지니스 로직 정의

## 작성한 이유
1. 문장, 글자 수, 토큰 등의 단위의 경우 맥락이 다수인 경우 임베딩으로 정확한 표현이 어려움
2. SBERT 모델을 활용하여 이전 문장과 현재 문장의 유사도를 비교하여 Chunk로 구성