根据对单一问题的验证，模型在100个step后就能成功的解决原本计算错误的能力，并且在600个step后居然会主动尝试写 python来验证自己的答案？(详见reslut_800steps.json)

同时对 COT数据集进行学习，尝试学习思维链，根据结果(result_14000steps.json)来看,只需要100个step，模型就能具有COT的生成结构

SFT的训练曲线：
![alt text](image.png)