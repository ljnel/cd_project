def check_custom_termination(gym_name: str, next_obs) -> bool:
    """Custom failure conditions beyond gymnasium defaults."""
    if gym_name == 'HalfCheetah-v5' and abs(next_obs[1]) > 0.9:
        return True
    return bool(gym_name == 'Ant-v5' and (not 0.2 <= next_obs[0] <= 1.0 or next_obs[1] < 0.5))
