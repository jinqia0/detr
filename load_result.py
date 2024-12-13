import torch

# 加载保存的评估文件
eval_data = torch.load("evaluate_results/eval.pth")

# 查看数据结构和键
print(eval_data.keys())

# # 检查具体内容
# print(eval_data['params'])
print(eval_data['precision'])
print(eval_data['recall'])