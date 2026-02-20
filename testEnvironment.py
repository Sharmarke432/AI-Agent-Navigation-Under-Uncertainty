import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from EnvironmentClass import GridWorldEnv

env=gym.make("GridWorld-v0")  # Create an instance of the environment

try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

locations,distance = env.reset(seed=42)  # Use seed for reproducible testing
print(f"Initial distance to goal: {distance}")  # Should be the distance from (0,0) to (size-1,size-1) for a 5x5 grid
print(f"Starting position - Agent: {locations['agent']}, Target: {locations['end']}")

# Test each action type
actions = [0, 1, 2, 3]  # right, up, left, down
new_pos = locations["agent"]
for action in actions:
    old_pos = new_pos.copy()
    newLoc, reward, terminated, truncated, info = env.step(action)
    new_pos = newLoc["agent"]
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")