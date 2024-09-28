import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.HumanRendering(env)
    env = gym.wrappers.RecordVideo(env, video_folder="video")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print(
        f"Episode finished in {total_steps} steps with total reward of {total_reward}"
    )
    env.close()
