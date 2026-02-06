# inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

class BERTEDLBinaryClassifier(nn.Module):
    def __init__(self, pretrained_model='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.5)
        self.evidence_layer = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        evidence = F.softplus(self.evidence_layer(pooled))
        return evidence

# 全局配置
DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = BertTokenizer.from_pretrained("bert-base-chinese")
ID2LABEL = {
    0: "原因类",
    1: "定义类",
    2: "建议类",
    3: "查询类",
    4: "结果类",
    5: "解决类"
}

# 加载模型（只执行一次）
def _load_models():
    print("正在加载6个二分类模型...")
    checkpoint = torch.load("final_multilabel_edl.pth", map_location=DEVICE, weights_only=False)
    models = []
    for i in range(checkpoint["num_classes"]):
        model = BERTEDLBinaryClassifier().to(DEVICE)
        model.load_state_dict(checkpoint["model_states"][i])
        model.eval()
        models.append(model)
    print("✅ 模型加载成功！")
    return models

LOADED_MODELS = _load_models()

def predict(text: str):
    """
    输入农业/通用问题文本，返回预测的类别列表
    示例:
        predict("桃树先开花还是先长叶？") → ["查询类"]
        predict("如何防治病虫害？") → ["建议类", "解决类"]
    """
    encoding = TOKENIZER(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(DEVICE)

    predicted_labels = []
    with torch.no_grad():
        for i, model in enumerate(LOADED_MODELS):
            evidence = model(encoding["input_ids"], encoding["attention_mask"])
            neg_evi = evidence[0, 0].item()
            pos_evi = evidence[0, 1].item()
            if pos_evi > neg_evi:
                predicted_labels.append(ID2LABEL[i])
    return predicted_labels