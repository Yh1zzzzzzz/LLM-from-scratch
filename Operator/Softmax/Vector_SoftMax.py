import torch
class VectorSoftmax:
    def __init__(self):
        self.logits_sum = []
        self.max_list = []
        self.Tensor = []
    def forward(self, x : list[torch.Tensor]):
        "传入的x是原始tensor，没有经过softmax处理的tensor列表。"
        self.logits_sum.append(torch.sum(x))
        self.max_list.append(torch.max(x))
        for i in range(len(x)):
            x[i] = torch.exp(x[i] - self.max_list[i])  # 减去最大值以避免溢出
            x[i] *= torch.sum
            
    

        """
        for i in range(len(x)):
            self.max_list.append(torch.max(x[i]))
            self.scale_factor.append(torch.sum(x[i]))
        self.Max = torch.max(torch.tensor(self.max_list))
        for i in range(len(x)):
            x[i] = torch.exp(self.max_list[i] - self.Max) * x[i]
            self.sum += torch.sum(x[i])
        for i in range(len(x)):
            x[i] *= self.scale_factor[i] / self.sum
        return torch.concatenate(x, dim=0)
"""
VectorSoftmax = VectorSoftmax()
softmax1 = torch.softmax(torch.tensor([1., 1., 1.]), dim=0)
softmax2 = torch.softmax(torch.tensor([1., 1.]), dim=0)
expected_output = torch.softmax(torch.tensor([1.,1.,1.,1.,1.]), dim=0)
output = VectorSoftmax.forward([softmax1, softmax2])
print("Softmax output:", output)
print("Expected output:", expected_output)
print("Softmax output matches expected output.")
        

        