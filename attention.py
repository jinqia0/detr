from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# 输入句子并编码
sentence = "The cat sat on the mat"
inputs = tokenizer(sentence, return_tensors='pt')

# 获取模型输出（包括注意力权重）
with torch.no_grad():
    outputs = model(**inputs)

# 提取注意力权重
attentions = outputs.attentions  # 获取注意力权重
# 取第一层注意力权重
attention_weights = attentions[0][0]  # Shape: [num_heads, sequence_length, sequence_length]

# 平均所有头部的注意力权重
attention_avg = attention_weights.mean(dim=0).detach().numpy()  # Shape: [sequence_length, sequence_length]

# 可视化注意力权重
plt.figure(figsize=(8, 6))
sns.heatmap(attention_avg, cmap="viridis", annot=True, fmt=".2f")
plt.xlabel("Key Sequence Position")
plt.ylabel("Query Sequence Position")
plt.title("BERT Self-Attention Heatmap")
plt.show()