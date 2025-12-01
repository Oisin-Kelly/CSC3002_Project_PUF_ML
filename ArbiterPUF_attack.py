from models.ArbiterPUF import ArbiterPUF
from helpers import get_XY_phi

arbiter_puf = ArbiterPUF(64)

# 1,000 challenges

challenges_responses = arbiter_puf.generate_challenges_reponses(1_000)

X, Y = get_XY_phi(challenges_responses)

arbiter_puf.train(X, Y)

# 10,000 challenges

challenges_responses = arbiter_puf.generate_challenges_reponses(10_000)

X, Y = get_XY_phi(challenges_responses)

arbiter_puf.train(X, Y)

# 100,000 challenges

challenges_responses = arbiter_puf.generate_challenges_reponses(100_000)

X, Y = get_XY_phi(challenges_responses)

arbiter_puf.train(X, Y)
