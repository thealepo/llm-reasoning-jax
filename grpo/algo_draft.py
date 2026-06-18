# iterative grpo (algo 1 in deepseekmath)

policy = PolicyModel()
reward = RewardModel()
data = ...

EPSILON = 0.2
BETA = 0.1
MU = ...

curr = nnx.state(policy)
for i in range(I):
    reference = curr
    for m in range(M):
        questions = batch(data)
        # update old policy model
        for q in questions:
            group = []
            rewards = []
            for g in range(G):
                output = curr.generate(q)
                group.append(output)
                r = reward(q,output)
                rewards.append(r)
            advantages = group_relative_advantages(rewards)

            for j in range(MU):
                