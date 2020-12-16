ray helper for distribute inf
---

1. 分布式inference，fifoqueue单独抽出来了，作为一个独立的数据模块。
2. 把Env和Model分别放到单独的Module中
3. 在Model上定义两个函数，build_learner, build_evaluator_model. 分别构建训练的模型和测试的模型
4. train.py向这些模块里面传Model和Env的构建函数及参数. 
