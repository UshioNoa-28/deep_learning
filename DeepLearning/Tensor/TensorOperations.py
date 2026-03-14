# 支持加减乘除
import torch
tensor=torch.tensor([1,2,3])
print(tensor*10)# 输出tensor([10, 20, 30])
# 基本的对标量的加减乘除都支持
# 需要注意的是矩阵乘法
tensor1=torch.tensor([1,2,3])
print(tensor1.shape)
tensor2=torch.tensor([[1],[2],[3]])
print(tensor1*tensor2) # 这个并非矩阵乘法
print(tensor1@tensor2)# 这个才是
# 为了矩阵更好看 推荐写为
tensor3=torch.tensor(
    [[1,3,4],
     [2,3,4],
     [1,2,34]],
)
# 可以通过tensor.T访问其转置矩阵
softmax_tensor=torch.tensor([1,2,3],dtype=torch.float16)
print(softmax_tensor.max())
print(softmax_tensor.min())
# 可以访问其最大最小值 在SoftMax回归中有用
print(softmax_tensor.mean()) # 还可以求平均值 不过需要注意的是默认是long类型 如果要求平均值需要手动指定为浮点数类型
# 同样在SoftMax回归中非常有用的是返回最大最小值的位置
print(softmax_tensor.argmax())
print(softmax_tensor.argmin())
# 输出为
# tensor(2)
# tensor(0)

# 可以像数组一样访问任意索引的元素 和矩阵一样支持:表示全部值

# 可复现性 如果你在每次随机生成矩阵之前都重置种子就能得到一样的结果 确保结果的可复现性
torch.manual_seed(42)
print(torch.rand(1,2))
torch.manual_seed(42)
print(torch.rand(1,2))

print(torch.cuda.is_available())


exercise_tensor=torch.rand(size=(7,7))
exercise1_tensor=torch.rand(size=(1,7))
print(exercise_tensor@exercise1_tensor.T)


test_tensor=torch.tensor([[1,2,3],[2,3,4]])
test_tensor.sum(dim=0) # 按列求和 也就是按行的方向叠加
test_tensor.sum(dim=1) # 按行求和 也就是按列的方向叠加
# 还提供一个额外参数 keepdim 一般而言 求和的时候会丢失该方向上的维度 keepdim表示保留该维度