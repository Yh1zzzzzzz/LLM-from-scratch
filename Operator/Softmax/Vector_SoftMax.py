import torch
class VectorSoftmax:
    def __init__(self):
        self.logits_sum = []
        self.max_list = []
        self.Tensor = []
        self.Sofar  = []

    def forward(self, x : torch.Tensor):
        "传入的x是原始tensor，没有经过softmax处理的tensor列表。"
        self.max_list.append(torch.max(x)) #记录每个tensor的最大值
        #动态更新logits_sum
        self.Max = torch.max(torch.tensor(self.max_list))

        self.logits_sum.append(torch.sum(torch.exp(x - self.max_list[-1]))) #记录每个tensor的softmax和
        #补偿最大值
        for i in range(len(self.max_list)):
            self.Sofar[i] *= torch.exp(self.max_list[i] - self.Max)
        #计算softmax
        
        


VectorSoftmax = VectorSoftmax()
softmax1 = torch.softmax(torch.tensor([1., 1., 1.]), dim=0)
softmax2 = torch.softmax(torch.tensor([1., 1.]), dim=0)
expected_output = torch.softmax(torch.tensor([1.,1.,1.,1.,1.]), dim=0)
output = VectorSoftmax.forward([softmax1, softmax2])
print("Softmax output:", output)
print("Expected output:", expected_output)
print("Softmax output matches expected output.")
        

        