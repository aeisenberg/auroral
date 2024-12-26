# Models

Reinforcement learning models to play the game. A model directory must contain an `__init__.py` file
that comprises a `model` class. Each class must implement the following methods:

- `step(state: np.ndarray, action: dict, reward: float, next_state: np.ndarray, done: bool) -> None`
  - `state`: Game frame
  - `action`: Action taken
  - `reward`: Reward received
  - `next_state`: Next game frame
  - `done`: Whether the episode ended
- `prepare_episode() -> None`: Reset internal state between episodes.
- `act(state: np.ndarray) -> dict`
  - `state`: Game frame
  - Return value: a dictionary formatted as
    `{"up": bool, "down": bool, "left": bool, "right", bool, "fire": bool}`
- `save(filepath: str) -> None`: Save the model to a file.
- `load(filepath: str) -> None`: Load the model

Index:

- `random`: Move and fire randomly (baseline model)
- `cnn`: Feed-forward convolutional neural network.
