import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

def compute_cosine_similarity(vector1, vector2):
    """ 코사인 유사도 계산 """
    return cosine_similarity([vector1], [vector2])[0][0]

def generate_chunks(sentences, model=None, tokenizer=None,
                    sim_threshold=0.65, overlap_threshold=0.4,
                    min_chunk_tokens=100, max_chunk_tokens=1000,
                    check_subtitle_maxlength=20, compare_sentence_size=1):
    """
    SBERT를 이용한 문장 기반 청크 생성 (Sliding Window 방식)

    - sentences: 입력 문장 리스트
    - model: SBERT 모델
    - tokenizer: 토크나이저
    - sim_threshold: 문장 병합 임계값
    - overlap_threshold: 청크 간 중첩 판단 임계값
    - min_chunk_tokens: 청크 최소 토큰 수
    - max_chunk_tokens: 청크 최대 토큰 수
    - check_subtitle_maxlength: 소제목일 경우의 최대 길이
    - compare_sentence_size: 비교할 문장 개수 (Sliding Window 크기)
    """

    if model is None:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    sentence_embeddings = model.encode(sentences)

    chunks = []  # 최종 청크 저장 리스트
    current_chunk = []  # 현재 청크에 포함된 문장
    current_chunk_embs = []  # 현재 청크 문장들의 임베딩
    current_chunk_tokens = 0  # 현재 청크의 토큰 수

    pattern = r'^<h[1-9]>'  # 소제목을 판별하는 패턴

    for idx, (sentence, emb) in enumerate(zip(sentences, sentence_embeddings)):
        additional_tokens = len(tokenizer.tokenize(sentence))

        # 청크 크기가 최대 토큰 수를 초과하면 새로운 청크 시작
        if current_chunk_tokens + additional_tokens > max_chunk_tokens:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_chunk_embs = [emb]
            current_chunk_tokens = additional_tokens
        else:
            if current_chunk:
                # 소제목 처리
                if (not re.match(pattern, sentence) and current_chunk_tokens <= min_chunk_tokens) or \
                        (re.match(pattern, sentence) and current_chunk_tokens <= check_subtitle_maxlength):
                    current_chunk.append(re.sub(pattern, '', sentence))
                    current_chunk_embs.append(emb)
                    current_chunk_tokens += additional_tokens
                else:
                    if re.match(pattern, sentence):
                        # 소제목은 새로운 청크로 처리
                        chunks.append(current_chunk)
                        current_chunk = [sentence]
                        current_chunk_embs = [emb]
                        current_chunk_tokens = additional_tokens
                    else:
                        # 여러 이전 문장들의 평균 임베딩을 활용한 유사도 비교
                        num_compare = max(1, min(compare_sentence_size, len(current_chunk_embs)))

                        if num_compare > 1:
                            prev_emb_avg = np.mean(current_chunk_embs[-num_compare:], axis=0)
                        else:
                            prev_emb_avg = current_chunk_embs[-1]

                        avg_similarity = compute_cosine_similarity(prev_emb_avg, emb)

                        # 유사도가 임계값 이상이면 현재 청크에 추가
                        if avg_similarity >= sim_threshold:
                            current_chunk.append(sentence)
                            current_chunk_embs.append(emb)
                            current_chunk_tokens += additional_tokens
                        else:
                            chunks.append(current_chunk)
                            current_chunk = [sentence]
                            current_chunk_embs = [emb]
                            current_chunk_tokens = additional_tokens
            else:
                current_chunk.append(sentence)
                current_chunk_embs.append(emb)
                current_chunk_tokens += additional_tokens

    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)

    # 청크 간 중첩 (Overlap) 보정
    adjusted_chunks = []
    for i in range(len(chunks)):
        if i > 0:
            prev_sentence = chunks[i - 1][-1]
            curr_sentence = chunks[i][0]

            prev_emb = model.encode([prev_sentence])[0]
            curr_emb = model.encode([curr_sentence])[0]
            overlap_sim = compute_cosine_similarity(prev_emb, curr_emb)

            # 소제목 태그 제거
            if re.match(pattern, curr_sentence):
                chunks[i][0] = re.sub(pattern, '', curr_sentence)
            elif overlap_sim >= overlap_threshold:
                chunks[i] = [prev_sentence] + chunks[i]

        # 중복 방지
        if len(adjusted_chunks) != 0 and set(adjusted_chunks[-1]).issubset(set(chunks[i])):
            adjusted_chunks[-1] = chunks[i]
        else:
            adjusted_chunks.append(chunks[i])

    return adjusted_chunks
