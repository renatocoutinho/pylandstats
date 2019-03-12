from .spatiotemporal import SpatioTemporalAnalysis

__all__ = ['MultiSpatioTemporalAnalysis']


class MultiSpatioTemporalAnalysis:
    def __init__(self, landscape_dict, **spatiotemporal_kws):
        """
        If the values of `landscape_dict` are lists of `Landscape` instances
        or lists of filepaths, `spatiotemporal_kws` will be passed to the
        constructor of `SpatioTemporalAnalysis`. Otherwise, i.e., if the
        values of `landscape_dict` are instances of `SpatioTemporalAnalysis`,
        `spatiotemporal_kws` will be ignored, and it is the user's
        responsibility to ensure that the corresponding instances of
        `SpatioTemporalAnalysis` are well constructed

        Parameters
        ----------
        landscape_dict : dict
            Dictionary mapping of a particular feature and its temporal
            sequence of landscapes, either in form of a list of landscapes
            (`Landscape` objects or file paths) or as a
            `SpatioTemporalAnalysis` instance.
        **spatiotemporal_kws : optional
            Keyword arguments to be passed to the constructor of
            `SpatioTemporalAnalysis` for each factor
        """

        if isinstance(
                next(iter(landscape_dict.values())), SpatioTemporalAnalysis):
            # `landscape_dict` is of the form
            # {'factor_1': SpatioTemporalAnalysis_1, ... }
            self.landscape_dict = landscape_dict
        else:  # or `if isinstance(landscape_dict_item, list)`
            # `landscape_dict` is of the form
            # {'factor_1': [Landscape_1^{t_1}, ... ], ... } or
            # {'factor_1': [filepath_to_landscape_1^{t_1}, ... ], ... }
            #
            self.landscape_dict = {
                factor: SpatioTemporalAnalysis(landscape_dict[factor],
                                               **spatiotemporal_kws)
                for factor in landscape_dict
            }

    def plot_metric(self, metric, class_val=None, ax=None, factor_legend=True,
                    plt_kws={}, **sta_plot_metric_kws):
        """
        All the arguments except `factor_legend` will be passed to the
        `plot_metric` method of the `SpatioTemporalAnalysis` instance that
        corresponds to each factor. See also the documentation of
        `SpatioTemporalAnalysis.plot_metric`.

        Parameters
        ----------
        metric : str
            A string indicating the name of the metric to plot
        class_val : int, optional
            If provided, the metric will be plotted at the level of the
            corresponding class, otherwise it will be plotted at the landscape
            level
        ax : axis object, optional
            Plot in given axis; if None creates a new figure
        factor_legend : bool, default True
            Whether the factor legend should be displayed
        plt_kws : dict
            Keyword arguments to be passed to `plt.plot`
        **sta_plot_metric_kws : optional
            Keyword arguments to be passed to `plot_metric` of each factor's
            `SpatioTemporalAnalysis` instance

        Returns
        -------
        ax : axis object
            Returns the Axes object with the plot drawn onto it
        """
        for factor, sta in self.landscape_dict.items():
            factor_plt_kws = plt_kws.copy()
            if factor_legend:
                factor_plt_kws['label'] = factor
            ax = sta.plot_metric(metric, class_val=class_val, ax=ax,
                                 plt_kws=factor_plt_kws, **sta_plot_metric_kws)
            if factor_legend:
                ax.legend()

        return ax

    def plot_metrics(self, class_val=None, metrics=None, metric_legend=True,
                     factor_legend=True, plt_kws={}, **sta_plot_metrics_kws):
        """
        Parameters
        ----------
        class_val : int, optional
            If provided, the metrics will be plotted at the level of the
            corresponding class, otherwise it will be plotted at the landscape
            level
        metrics : list-like, optional
            A list-like of strings with the names of the metrics that should
            be plotted. The metrics should have been passed within the
            initialization of this `SpatioTemporalAnalysis` instance
        metric_legend : bool, default True
            Whether the metric label should be displayed within the plot (as
            label of the y-axis)
        factor_legend : bool, default True
            Whether the factor legend should be displayed
        plt_kws : dict
            Keyword arguments to be passed to `plt.plot`
        **sta_plot_metrics_kws : optional
            Keyword arguments to be passed to `plot_metrics` of the first
            factor's `SpatioTemporalAnalysis` instance

        Returns
        -------
        fig, ax : tuple
            - figure object
            - axis object with the plot drawn onto it
        """

        # let's use the head-tail design pattern
        landscape_dict_iter = iter(self.landscape_dict.items())
        # get the `SpatioTemporalAnalysis` instance for the first factor
        factor, sta = next(landscape_dict_iter)
        # extract its metrics
        if metrics is None:
            if class_val is None:
                metrics = sta.landscape_metrics
            else:
                metrics = sta.class_metrics
        # plot it
        factor_plt_kws = plt_kws.copy()
        if factor_legend:
            factor_plt_kws['label'] = factor
        fig, axes = sta.plot_metrics(
            class_val=class_val, metrics=metrics, metric_legend=metric_legend,
            plt_kws=factor_plt_kws, **sta_plot_metrics_kws)
        # get flat axes
        if len(axes) == 1:
            if len(axes[0]) == 1:
                flat_axes = [axes]
        else:
            flat_axes = axes.flatten()

        # now plot the rest
        for factor, sta in landscape_dict_iter:
            factor_plt_kws = plt_kws.copy()
            if factor_legend:
                factor_plt_kws['label'] = factor

            for metric, ax in zip(metrics, flat_axes):
                ax = sta.plot_metric(metric, class_val=class_val, ax=ax,
                                     metric_legend=metric_legend,
                                     plt_kws=factor_plt_kws)
                if factor_legend:
                    ax.legend()

        return fig, axes
