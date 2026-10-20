from src.OK_dqn.agent.random_agent import RandomAgent
from src.OK_dqn.env.cart_pole import make_cart_pole
from src.training.dqn_harness import train

my_agent = RandomAgent(make_cart_pole().action_space)
train(my_agent, make_cart_pole, 10000, group="OK", name="OK")
