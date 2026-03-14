import torch
# 张量 标量是1阶张量 向量是2阶 矩阵是3阶
tensor=torch.tensor([1,1]) # 定义一个向量
print(tensor.ndim) #访问该张量的维数 输出1
print(tensor.shape)#访问该张量的形状 输出torch.Size([2])
MATRIX=torch.tensor([[1,2],[2,3]])# 一般而言小写表示标量和向量 大写表示矩形和高阶张量
print(MATRIX.shape) #输出torch.Size([2, 2])
# 一般而言 在进行学习的时候会带学习的参数提供一个初始值 你可以使用
print(torch.rand(size=(3,4))) #生成一个随机的 3行 4列 的矩阵
torch.zeros(size=(3,4)) #创建一个全0的大小为3 4的矩阵
torch.ones(size=(4,3)) #同上 只不过是全为1
# 如果你想要创建相同大小的矩阵且全为0或者1
torch.zeros_like(MATRIX)
# 你还可以在创建张量的时候指定张量的类型 torch会根据张量的类型来判断在cpu上还是gpu上进行计算
float_16_tensor=torch.tensor([1,2,3],dtype=torch.float16) # 这样即可创建一个指定类型的张量
# 一般而言张量最重要的属性也在上面提及了 Shape Type Device(cpu,gpu)
print(float_16_tensor.device) # 输出cpu
float_16_tensor=float_16_tensor.to("cuda")# 将其搬到gpu上
print(float_16_tensor)