import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class MySwarmPlotter(sns.categorical._SwarmPlotter):

    def plot(self, ax, inlier, kws):
        """Make the full plot."""
        self.draw_swarmplot(ax, inlier, kws)
        self.add_legend_data(ax)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()
    
    def draw_swarmplot(self, ax, inlier, kws):
        """Plot the data."""
        s = kws.pop("s")

        centers = []
        swarms = []

        # Set the categorical axes limits here for the swarm math
        if self.orient == "v":
            ax.set_xlim(-.5, len(self.plot_data) - .5)
        else:
            ax.set_ylim(-.5, len(self.plot_data) - .5)

        # Plot each swarm
        for i, group_data in enumerate(self.plot_data):

            if self.plot_hues is None or not self.split:

                width = self.width

                if self.hue_names is None:
                    hue_mask = np.ones(group_data.size, np.bool)
                else:
                    hue_mask = np.array([h in self.hue_names
                                         for h in self.plot_hues[i]], np.bool)
                    # Broken on older numpys
                    # hue_mask = np.in1d(self.plot_hues[i], self.hue_names)

                swarm_data = group_data[hue_mask]

                # Sort the points for the beeswarm algorithm
                sorter = np.argsort(swarm_data)
                swarm_data = swarm_data[sorter]
                point_colors = self.point_colors[i][hue_mask][sorter]

                # Plot the points in centered positions
                cat_pos = np.ones(swarm_data.size) * i
                kws.update(c=point_colors)
                if self.orient == "v":
                    sns.set_context(rc={'lines.markeredgewidth': 1})
                    ax.scatter(cat_pos[~inlier[i]], swarm_data[~inlier[i]],
                               marker=r'$\mathbf{\times}$', s=5*s, **kws)
                    sns.set_context(rc={'lines.markeredgewidth': 0})
                    points = ax.scatter(cat_pos[inlier[i]], swarm_data[inlier[i]], s=s, **kws)

                else:
                    points = ax.scatter(swarm_data, cat_pos, s=s, **kws)

                centers.append(i)
                swarms.append(points)

            else:
                offsets = self.hue_offsets
                width = self.nested_width

                for j, hue_level in enumerate(self.hue_names):
                    hue_mask = self.plot_hues[i] == hue_level
                    swarm_data = group_data[hue_mask]

                    # Sort the points for the beeswarm algorithm
                    sorter = np.argsort(swarm_data)
                    swarm_data = swarm_data[sorter]
                    point_colors = self.point_colors[i][hue_mask][sorter]

                    # Plot the points in centered positions
                    center = i + offsets[j]
                    cat_pos = np.ones(swarm_data.size) * center
                    kws.update(c=point_colors)
                    if self.orient == "v":
                        points = ax.scatter(cat_pos, swarm_data, s=s, **kws)
                    else:
                        points = ax.scatter(swarm_data, cat_pos, s=s, **kws)

                    centers.append(center)
                    swarms.append(points)

        # Update the position of each point on the categorical axis
        # Do this after plotting so that the numerical axis limits are correct
        for center, swarm in zip(centers, swarms):
            if swarm.get_offsets().size:
                self.swarm_points(ax, swarm, center, width, s, **kws)
                
                
def myswarmplot(x=None, y=None, hue=None, data=None, order=None,
                hue_order=None, split=False, orient=None, color=None,
                palette=None, size=5, edgecolor="gray", linewidth=0, ax=None,
                inlier=None, **kwargs):
    
    plotter = MySwarmPlotter(x, y, hue, data, order, hue_order, split, orient,
                             color, palette)
    
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault("zorder", 3)
    size = kwargs.get("s", size)
    if linewidth is None:
        linewidth = size / 10
    if edgecolor == "gray":
        edgecolor = plotter.gray
    kwargs.update(dict(s=size ** 2,
                       edgecolor=edgecolor,
                       linewidth=linewidth))

    plotter.plot(ax, inlier, kwargs)
    
    return ax, plotter
