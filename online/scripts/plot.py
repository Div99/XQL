import argparse
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

LOG_FILE_NAME = "log.csv"

def moving_avg(x, y, window_size=1):
    if window_size == 1:
        return x, y
    moving_avg_y = np.convolve(y, np.ones(window_size) / window_size, 'valid') 
    return x[-len(moving_avg_y):], moving_avg_y

def plot_run(paths, name, ax=None, x_key="steps", y_keys=["eval/loss"], window_size=1, max_x_value=None, **kwargs):
    for path in paths:
        assert LOG_FILE_NAME in os.listdir(path), "Did not find log file, found " + " ".join(os.listdir(path))
    for y_key in y_keys:
        xs, ys = [], []
        for path in paths:
            df = pd.read_csv(os.path.join(path, LOG_FILE_NAME))
            if y_key not in df:
                print("[research] WARNING: y_key was not in run, skipping plot", path)
            x, y = moving_avg(df[x_key], df[y_key], window_size=window_size)
            assert len(x) == len(y)
            if max_x_value is not None:
                y = y[x <= max_x_value] # need to set y value first
                x = x[x <= max_x_value]
            xs.append(x)
            ys.append(y)
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        plot_df = pd.DataFrame({x_key: xs, y_key: ys})
        label = name + " " + y_key if len(y_keys) > 1 else name
        ci = "sd" if len(paths) > 0 else None
        sns.lineplot(ax=ax, x=x_key, y=y_key, data=plot_df, sort=True, ci=ci, label=label, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="plot.png", help="Path of output plot")
    parser.add_argument("--path", "-p", nargs='+', type=str, required=True, help="Paths of runs to plot")
    parser.add_argument("--legend", "-l", nargs='+', type=str, required=False, help="Names of each run to display in the legend")
    parser.add_argument("--title", "-t", type=str, required=False, help="Plot title")
    parser.add_argument("--window", "-w", type=int, default=1, help="Moving window averaging parameter.")
    parser.add_argument("--x", "-x", type=str, default="step", help="X value to plot")
    parser.add_argument("--max-x", "-m", type=int, default=None, help="Max x value to plot")
    parser.add_argument("--x-label", "-xl", type=str, default=None, help="X label to display on the plot")
    parser.add_argument("--y", "-y", type=str, nargs='+', default=["eval/loss"], help="Y value(s) to plot")
    parser.add_argument("--y-label", "-yl", type=str, default=None, help="Y label to display on the plot")
    parser.add_argument("--fig-size", "-f", nargs=2, type=int, default=(6, 4))
    args = parser.parse_args()

    paths = args.path
    # Check to see if we should auto-expand the path.
    # Do this only if the number of paths specified is one and each sub-path is a directory
    if len(paths) == 1 and all([os.path.isdir(os.path.join(paths[0], d)) for d in os.listdir(paths[0])]):
        paths = sorted([os.path.join(paths[0], d) for d in os.listdir(paths[0])])
    # Now create the labels
    labels = args.legend
    if labels is None:
        labels = [os.path.basename(path[:-1] if path.endswith('/') else path) for path in paths]
    # Sort the paths alphabetically by the labels
    paths, labels = zip(*sorted(zip(paths, labels), key=lambda x: x[0])) # Alphabetically sort by filename
    
    for path, label in zip(paths, labels):
        if LOG_FILE_NAME not in os.listdir(path):
            path = [os.path.join(path, run) for run in os.listdir(path)]
        else:
            path = [path]
        sns.set_context(context="paper", font_scale=1.2)
        sns.set_style("darkgrid", {'font.family': 'serif'})
        plot_run(path, label, x_key=args.x, y_keys=args.y, window_size=args.window, max_x_value=args.max_x)
    
    # Set relevant labels
    if args.title:
        plt.title(args.title)
    # Label X
    if args.x_label is not None:
        plt.x_label(args.xlabel)
    elif args.x is not None:
        plt.xlabel(args.x)
    # Label Y
    if args.y_label is not None:
        plt.ylabel(args.y_label)
    elif args.y is not None and len(args.y) == 1:
        plt.ylabel(args.y[0])
    
    # Save the plot
    print("[research] Saving plot to", args.output)
    plt.gcf().set_size_inches(*args.fig_size)
    plt.tight_layout(pad=0)
    plt.savefig(args.output, dpi=200) # Increase DPI for higher res.
