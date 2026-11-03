from coppercore.dqn.snake_env import SnakeEnv
from coppercore.training.dqn_harness import train
from ok_dqn.agent.Q_agent import QAgent


def make_snake_env():
    return SnakeEnv(width=5, height=5, seed=0, max_steps_no_food=100)


def main():
    env = make_snake_env()
    ql_agent = QAgent(env.n_states, env.n_actions)

    train(
        ql_agent,
        make_snake_env,
        150_000,
        eval_every=2_000,
        name="OK",
        config={
            "alpha": ql_agent.alpha,
            "gamma": ql_agent.gamma,
            "epsilon_decay": ql_agent.epsilon_decay,
        },
    )


if __name__ == "__main__":
    main()
