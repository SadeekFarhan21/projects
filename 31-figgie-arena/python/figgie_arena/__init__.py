"""Figgie Arena: exact Figgie rules engine (C++20) with Python bindings.

The compiled extension ``_figgie`` is built by CMake into this directory.
"""

from ._figgie import (  # noqa: F401
    ActType,
    Config,
    Game,
    Status,
    VecEnv,
    all_configs,
    bot_names,
    config_posterior,
    constraint_probability,
    fair_values,
    fuzz,
    goal_probs,
    run_games,
)

SUITS = ("spades", "clubs", "hearts", "diamonds")
