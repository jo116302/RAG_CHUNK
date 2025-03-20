import re

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

def compute_cosine_similarity(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

def generate_chunks(sentences, model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'), tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'),
                    sim_threshold=0.65, overlap_threshold=0.4, min_chunk_tokens=100, max_chunk_tokens=1000, check_subtitle_maxlength=20, compare_sentence_size=1):
    """
    sentences: 분할된 문장들 리스트
    model: 이전에 로드한 SentenceTransformer 모델 (KoSBERT)
    tokenizer: 이전에 로드한 BertTokenizer
    sim_threshold: 인접 문장 간 병합 임계치
    overlap_threshold: 청크 간 중첩(overlap) 판단 임계치
    max_chunk_tokens: 청크 당 최대 토큰 수
    """
    # 문장 임베딩 생성
    print(sentences)
    sentence_embeddings = model.encode(sentences)

    chunks = []  # 최종 청크들을 저장할 리스트
    current_chunk = []  # 현재 청크 구성 중인 문장들 저장
    current_chunk_embs = []  # 현재 청크에 해당하는 문장 임베딩 정보 저장
    current_chunk_tokens = 0  # 현재 청크의 토큰 수

    pattern = r'^<h[1-9]>'

    for idx, (sentence, emb) in enumerate(zip(sentences, sentence_embeddings)):
        additional_tokens = len(tokenizer.tokenize(sentence))

        if current_chunk_tokens + additional_tokens > max_chunk_tokens:
            # 청크의 최대 토큰 수를 초과할 경우, 새로운 청크로 넘어갑니다.
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_chunk_embs = [emb]
            current_chunk_tokens = additional_tokens
        else:
            if current_chunk:
                if ((not re.match(pattern, sentence)) and current_chunk_tokens <= min_chunk_tokens) or (re.match(pattern, sentence) and current_chunk_tokens <= check_subtitle_maxlength):
                    current_chunk.append(re.sub(pattern, '', sentence))
                    current_chunk_embs.append(emb)
                    current_chunk_tokens += additional_tokens
                else:
                    if re.match(pattern, sentence):
                        # 새로운 청크로 변경
                        chunks.append(current_chunk)
                        current_chunk = [sentence]
                        current_chunk_embs = [emb]
                        current_chunk_tokens = additional_tokens
                    else:
                        # 여러 이전 문장들과 현재문장의 평균 유사도 계산
                        num_compare = min(compare_sentence_size, len(current_chunk_embs))
                        similarities = [compute_cosine_similarity(prev_emb, emb) for prev_emb in current_chunk_embs[-num_compare:]]
                        avg_similarity = sum(similarities) / len(similarities)

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

    if current_chunk:
        chunks.append(current_chunk)

    adjusted_chunks = []
    for i in range(len(chunks)):
        if i > 0 :
            prev_sentence = chunks[i - 1][-1]
            curr_sentence = chunks[i][0]
            prev_emb = model.encode([prev_sentence])[0]
            curr_emb = model.encode([curr_sentence])[0]
            overlap_sim = compute_cosine_similarity(prev_emb, curr_emb)

            if re.match(pattern, curr_sentence):
                chunks[i][0] = re.sub(pattern, '', curr_sentence)
            elif overlap_sim >= overlap_threshold:
                chunks[i] = [prev_sentence] + chunks[i]

        if len(adjusted_chunks) != 0 and set(adjusted_chunks[-1]).issubset(set(chunks[i])):
            adjusted_chunks[-1] = chunks[i]
        else:
            adjusted_chunks.append(chunks[i])

    return adjusted_chunks

