import gymnasium as gym

# from gymnasium.utils import play

if __name__ == "__main__":
    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.make("CartPole-v1")
    total_reward, total_steps = 0.0, 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated or truncated:
            break
    print(f"Episode finished in {total_steps} steps with a score of {total_reward:.2f}")
    env.close()
    # play.play(env, keys_to_action={'j': 0, 'k': 1})
