import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog="graph.py",
    description="Graph performances."
)
parser.add_argument(
    "path",
    help="Filepath to the evaluation file.",
    type=str
)
args = parser.parse_args()

with open(args.path) as f:
    content = json.load(f)

averages = []
steps = []
for level in content:
    averages.append(level["average_score"])
    steps.append(level["average_n_steps"])

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax3 = ax.twinx()
x = [i * 10 for i in range(len(averages))]
ax2.plot(x, averages)
# ax3.plot(x, steps)
ax.grid()
plt.show()
