Auroral
=======

- `English (en) <#A-self-contained-Game-for-Reinforcement-Learning>`_
- `Français (fr) <#Un-jeu-2D-pour-l'apprentissage-par-renforcement>`_


A Self-Contained Game to Explore Reinforcement Learning
-------------------------------------------------------

2D reinforcement learning environment to test machine learning models. You can use the environment
to train agents or just play the game ``:)``.


Installation
````````````

Execute the following commands to install the environment and dependencies:

.. code-block:: bash

   git clone git@github.com:Vincent-Therrien/auroral.git  # Download the repository.
   cd auroral  # Navigate inside of the repository.
   pip install basic_requirements.txt  # If you just want to play.
   pip install rl_requirements.txt  # If you want train and use reinforcement learning agents.


Usage
`````

**Play** the game:

.. code-block:: bash

   python3 play.py  # Linux
   py play.py  # Windows

**Train** a reinforcement learning agents:

.. code-block:: bash

   python3 train.py <configuration file> --output <output file>  # Linux
   py train.py <configuration file> --output <output file>  # Windows

The ``<configuration file>`` is a JSON file that parametrizes the training session. You can use,
for example, the file ``training/test.json``. The ``<output file>`` is an optional parameter. It is
used to save the trained model to a file.


Un jeu 2D pour l'apprentissage par renforcement
-----------------------------------------------
