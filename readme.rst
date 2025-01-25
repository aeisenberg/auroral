Auroral
=======

- `English (en) <#A-self-contained-Game-for-Reinforcement-Learning>`_
- `Français (fr) <#Jeux-2D-pour-explorer-l'apprentissage-par-renforcement>`_

.. image:: demo.gif
   :width: 600
   :alt: Comparison of the models. On the left, the untrained model scores 1 point over 12 seconds
      while on the right, the trained model scores 14 points in the same time frame.


Games to Explore Reinforcement Learning
---------------------------------------

2D reinforcement learning environment to test reinforcement learning models. You can use the
environment to train agents or just play the game ``:)``.


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

   python3 train.py <configuration file> --output <output directory>  # Linux
   py train.py <configuration file> --output <output directory>  # Windows

The ``<configuration file>`` is a JSON file that parametrizes the training session. You can use,
for example, the file ``training/test.json``. The ``<output directory>`` is an optional parameter.
It is used to save the trained model to a file.

**Test** the models:

.. code-block:: bash

   python3 test.py <configuration directory>  # Linux
   py test.py <configuration directory>  # Windows

The ``<configuration directory>`` is the ``<output directory>`` provided to the last command. The
repository already contains a trained model, so you can run, for instance:

.. code-block:: bash

   python3 test.py trained_models/dqn2  # Linux
   py test.py trained_models\dqn2  # Windows


Jeux 2D pour explorer l'apprentissage par renforcement
------------------------------------------------------
