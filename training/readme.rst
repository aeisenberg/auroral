Training Configuration Files
============================

The configuration file provided to the script must have the following content:

```
{
    "name": <Name of the model stored in the "models" directory to use>,
    "maximum_n_steps": <Maximum umber of steps in on episode>,
    "n_episodes": <Maximum number of episodes>,
    "framerate": <Number of frames per seconds (e.g. 30)>,
    "levels": [
        {
            "frequency": <Number between 0 and 1 indicating how frequently this
                level should be used for training.>,
            "theme": <Name of the level's theme>,
            "n": <Dimension of the level (int)>,
            "points": <Number of points (either int or pair of ints)>,
            "walls": <Number of walls (either int or pair of ints)>,
            "water": <Number of water tiles (either int or pair of ints)>,
            "trees": <Number of trees (either int or pair of ints)>,
            "doors": <Number of doors (either int or pair of ints)>,
            "enemies": <Number of enemies (either int or pair of ints)>,
            "danger": <Number of danger tiles (either int or pair of ints)>
        }
    ]
}
```
