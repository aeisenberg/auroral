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
for level in content:
    averages.append(level["average_score"])

fig, ax = plt.subplots()
x = [i * 10 for i in range(len(averages))]
ax.plot(x, averages)
ax.set(xlabel='Level', ylabel='Score',
       title='Score over levels')
ax.grid()
plt.show()
