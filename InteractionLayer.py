from torch import nn
from AttentionLayer import *
from transformers import ModernBertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch


class TokenMatchLayer(nn.Module):
    """Source-Hypothesis 토큰 간 차이를 감지하는 레이어"""
    def __init__(self, d_model, sep_token_id):
        super().__init__()
        self.sep_token_id = sep_token_id
        
        # 토큰 차이를 요약하는 attention
        #self.ddl_attn = DDL_ATTN(d_model, n_heads=8)

        self.diff_norm = nn.LayerNorm(d_model)
        self.diff_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.ddl_attn = AttentionLayer(d_model= d_model,d_ff = d_model, n_heads=8, drop_out=0.2, loops=1)



    def forward(self, hidden_states, input_ids, attention_mask):
        """
        hidden_states: [batch, seq_len, d_model]
        input_ids: [batch, seq_len] — SEP 위치 찾기용
        """
        batch_size, seq_len, d_model = hidden_states.shape  # 현재 shape가지고 옴
        output_states = hidden_states.clone() # 새로운 state를 담을 변수
        q = self.query_prams
        # [SEP] 위치 찾아서 source / hypothesis 분리
        all_match_feats = []
        for b in range(batch_size): #현재 배치들을 훑음

            #sep토큰 위치 찾기
            sep_mask = torch.eq(input_ids[b], self.sep_token_id)
            sep_positions = sep_mask.nonzero(as_tuple=True)[0]
            
            if len(sep_positions) >= 2:
                # source: [CLS] 다음 ~ 첫 번째 [SEP] 전
                src_start, src_end = 1, sep_positions[0].item() #source 토큰 위치
                # hypothesis: 첫 번째 [SEP] 다음 ~ 두 번째 [SEP] 전  
                hyp_start, hyp_end = src_end + 1, sep_positions[1].item() #hypothesis 위치
            else:
                # fallback: 절반으로 나누기 SEP토큰이 1개이하라 아예 토큰 내용을 반으로 잘라서 서로 attention
                mid = seq_len // 2
                src_start, src_end = 1, mid
                hyp_start, hyp_end = mid, seq_len - 1
            
            #현재 배치번호에서의 source와 hypothesis의 토큰들
            src_tokens = hidden_states[b, src_start:src_end]   # [src_len, d]
            hyp_tokens = hidden_states[b, hyp_start:hyp_end]   # [hyp_len, d]
            

            if src_tokens.size(0) == 0 or hyp_tokens.size(0) == 0: #source나 hypo가 없을때 처리.
                all_match_feats.append(torch.zeros(d_model, device=hidden_states.device)) #영벡터 추가후 continue로 다음 배치로 넘어감
                continue
            
            # hyp→src cross-attention: hypothesis의 각 토큰이 source에서 가장 관련 있는 부분을 찾고, 그 "차이"를 계산

            
            updated_hyp = self.ddl_attn(hyp_tokens, src_tokens)  #요약문은 query, 본문은 key, value로 사용해서 일종의 encoder-decoder attention

            # updated_hyp: source에서 hypothesis와 매칭되는 부분
            
            
            diff_raw  = updated_hyp
            diff = self.diff_norm(diff_raw ) #이 토큰이 어떻게 다른지

            padding_diff = diff.squeeze(0)
            output_states[b, hyp_start:hyp_end] = hyp_tokens + padding_diff

            #all_match_feats.append(pooled)#(pooled_raw + pooled_normed)
        
        return output_states
        #match_feats = torch.stack(all_match_feats)  # [batch, d_model]
        #return match_feats + self.diff_pool(match_feats)



class ModernBertWithTokenMatch(ModernBertForTokenClassification):
    """full hidden states를 활용하는 커스텀 모델"""
    
    def set_token_match(self, sep_token_id):
        d_model = self.config.hidden_size  # 1024
        self.token_match = TokenMatchLayer(d_model, sep_token_id)
        self.match_feat_norm = nn.LayerNorm(d_model) 
        # 토큰 분류에서는 차원을 늘렸다가 다시 줄이는 merge 층이 필요없습니다.
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # 1. BERT 인코딩 (전체 hidden states)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]  # [batch, seq_len, d_model]
        
        # 2. 토큰 매칭 특징 interaction layer 
        # [batch, seq_len, d_model] 차원이 유지되어 반환됩니다.
        match_feat = self.token_match(hidden_states, input_ids, attention_mask)
        match_feat = self.match_feat_norm(match_feat) 
        
        # 3. 기존 classifier
        # 시퀀스 전체를 통과시켜 [batch, seq_len, num_labels] 형태로 추출
        logits = self.classifier(match_feat)
        
        loss = None
        if labels is not None:
            # Token Classification용 Loss 계산
            # 보통 패딩 토큰은 라벨을 -100으로 설정하므로, 이를 무시하도록 설정합니다.
            if self.num_labels == 1:
                # num_labels가 1일 때 (BCE)
                loss_fct = nn.BCEWithLogitsLoss()
                active_loss = labels.view(-1) != -100
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss].float()
                loss = loss_fct(active_logits, active_labels)
            else:
                # num_labels > 1 일 때 (CE)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        # TokenClassifierOutput으로 반환
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


