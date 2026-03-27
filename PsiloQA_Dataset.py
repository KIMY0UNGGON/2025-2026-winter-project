
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset



#단어 라벨을 토큰 라벨로 변경해주는 함수
def align_labels_to_tokens(source, question , words, word_labels, tokenizer, max_length=512):
    """
    source: 지문 (String)
    question: 질문 (String)
    words: ["이순신은", "조선의", "장군이다", "."] (요약문/답변 단어 리스트)
    word_labels: [0, 0, 1, 0] (각 단어에 대한 정답 라벨)
    """
    # 1. 첫 번째 그룹(Context): 지문과 질문을 하나로 합친 뒤 단어 단위(리스트)로 쪼갭니다.
    context_text = f"{question} {source}"
    context_words = context_text.split()
    
    # 2. 토크나이저에 2개의 그룹으로 넘겨줍니다.
    # text = 첫 번째 그룹 (질문+지문)
    # text_pair = 두 번째 그룹 (요약문/답변)
    tokenized = tokenizer(
        text=context_words, 
        text_pair=words, 
        is_split_into_words=True, 
        truncation="only_first",  # 전체 길이가 max_length 초과 시 첫 번째 그룹(지문/질문)을 먼저 자름
        max_length=max_length
    )
    
    # 3. 매핑 로직: sequence_ids()를 사용해 토큰의 출처 구분!
    word_ids = tokenized.word_ids()
    sequence_ids = tokenized.sequence_ids()
    
    aligned_labels = []
    previous_word_idx = None
    
    for word_idx, seq_idx in zip(word_ids, sequence_ids):
        # seq_idx == 0: 첫 번째 그룹 (질문+지문)에서 옴
        # seq_idx == 1: 두 번째 그룹 (요약문/답변)에서 옴
        # seq_idx == None: 특수 토큰 ([CLS], [SEP] 등)
        
        if seq_idx != 1:
            # 특수 토큰이거나 질문/지문 구간인 경우 -> 환각 정답 검사에서 제외 (-100)
            aligned_labels.append(-100)
            previous_word_idx = None
            
        else:
            # 요약문/답변 구간인 경우 -> 원본 단어의 라벨을 매핑
            if word_idx != previous_word_idx:
                aligned_labels.append(word_labels[word_idx])
            else:
                aligned_labels.append(word_labels[word_idx]) 
            previous_word_idx = word_idx
            
    tokenized["labels"] = aligned_labels
    return tokenized


    

def load_dataset(split = "train"):

    if split != "train" or split != "test" or split != "validation" or split != "valid":
        raise ValueError("split must be train, test, or validation")

        
    with open(f"dataset/token_data_{split}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data



class TokenHalDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, split="train"):
        samples = load_dataset(split)
        self.features = []
        

        # tqdm을 씌워 진행 과정을 볼 수 있게 합니다.
        for s in tqdm(samples, desc=f"Preprocessing split {split}"):
            
            # TODO: 실제 JSON 파일의 키(key) 이름으로 맞춰주세요.
            # 단어별로 쪼개진 배열 ("사과가", "나무에서", ...)
            words = s.get("words", [])       
            
            # 각 단어에 매핑된 라벨 배열 (0, 0, 1, ...)
            word_labels = s.get("labels", [])  
            
            wiki_passage = s.get("wiki_passage" , "")
            question = s.get("question", "")
            # 데이터가 비어있지 않고, 단어 수와 라벨 수가 일치하는지 확인
            if len(words) > 0 and len(words) == len(word_labels):
                
                # 1. 위에서 만든 함수로 토큰화 & 라벨 정렬!
                tokenized = align_labels_to_tokens(wiki_passage, question, words, word_labels, tokenizer, max_length)
                
                # 2. 결과 딕셔너리에서 필요한 텐서 배열들을 뽑아서 features에 저장
                self.features.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["labels"],
                    "wiki_passage": wiki_passage,
                    "question": question
                })
        
    def __len__(self): 
        return len(self.features)
        
    def __getitem__(self, idx):
        item = self.features[idx]
        return_item = {}
        
        for k, v in item.items():
            if isinstance(v, str):
                return_item[k] = v
            else:
                return_item[k] = torch.tensor(v)
                
        return return_item
