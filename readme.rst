Auroral
=======

- `English (en) <#Games-to-Explore-Reinforcement-Learning>`_
- `Français (fr) <#jeux-2D-pour-explorer-lapprentissage-par-renforcement>`_

.. image:: assets/demo.gif
   :width: 400
   :align: center
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

Two games are available. You can select them with the ``--game <1 or 2>`` command-line argument.
First game:

.. image:: assets/game1.png
   :align: middle
   :width: 200

Select a level with the command-line option ``--level <n>``, where ``n`` is between 1 and 11,
inclusively. Second game:

.. image:: assets/game2.png
   :align: middle
   :width: 200

**Train** a reinforcement learning agents:

.. code-block:: bash

   python3 train.py <configuration file> --output <output directory>  # Linux
   py train.py <configuration file> --output <output directory>  # Windows

The ``<configuration file>`` is a JSON file that parametrizes the training session. You can use,
for example, the file ``training/dqn2.json``. The ``<output directory>`` is an optional parameter.
It is used to save the trained model.

**Test** a model:

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

