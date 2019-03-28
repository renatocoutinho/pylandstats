import numpy as np
import rasterio
from rasterio import features

from .landscape import Landscape
from .multilandscape import MultiLandscape

try:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry
    geo_imports = True
except ImportError:
    geo_imports = False

__all__ = ['GradientAnalysis', 'BufferAnalysis']


class GradientAnalysis(MultiLandscape):
    def __init__(self, landscape, masks_arr, feature_name=None,
                 feature_values=None, **kwargs):
        if not isinstance(landscape, Landscape):
            landscape = Landscape(landscape)

        landscapes = [
            Landscape(
                np.where(mask_arr, landscape.landscape_arr, landscape.nodata),
                res=(landscape.cell_width,
                     landscape.cell_height), nodata=landscape.nodata)
            for mask_arr in masks_arr
        ]

        # TODO: is it useful to store `masks_arr` as instance attribute?
        self.masks_arr = masks_arr

        # The feature name will be `buffer_dists` for `BufferAnalysis` or
        # `transect_dist` for `TransectAnalysis`, but for any other custom use
        # of `GradientAnalysis`, the user might provide (or not) a custom name
        if feature_name is None:
            feature_name = 'feature_values'

        # If the values for the distinguishing feature are not provided, a
        # basic enumeration will be automatically generated
        if feature_values is None:
            feature_values = [i for i in range(len(masks_arr))]

        # now call the parent's init
        super(GradientAnalysis, self).__init__(landscapes, feature_name,
                                               feature_values, **kwargs)

    def plot_metric(self, metric, class_val=None, ax=None, metric_legend=True,
                    method='bar', method_kws={}, fmt='--o', subplots_kws={}):
        """
        Override of `MultiLandscape.plot_metric` in order to change the
        default value for the `method` argument to 'bar' (instead of 'plot')

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
        metric_legend : bool, default True
            Whether the metric label should be displayed within the plot (as
            label of the y-axis)
        method : string, default 'bar'
            The matplotlib method to use for the plot. Options are: 'plot' or
            'bar'
        method_kws : dict
            Keyword arguments to be passed to the matplotlib plotting method
            as indicated in the `method` argument
        fmt : str, default '--o'
            A format string for the `plt.plot`. Ignored if `method` is 'bar'
        subplots_kws : dict
            Keyword arguments to be passed to `plt.subplots`, only if no axis
            is given (through the `ax` argument)

        Returns
        -------
        ax : axis object
            Returns the Axes object with the plot drawn onto it
        """

        return super(GradientAnalysis, self).plot_metric(
            metric, class_val=class_val, ax=ax, metric_legend=metric_legend,
            method=method, method_kws=method_kws, fmt=fmt,
            subplots_kws=subplots_kws)

    def plot_metrics(self, class_val=None, metrics=None, num_cols=3,
                     metric_legend=True, xtick_labelbottom=False, method='bar',
                     method_kws={}, fmt='--o', subplots_adjust_kws={}):
        """
        Override of `MultiLandscape.plot_metrics` in order to change the
        default value for the `method` argument to 'bar' (instead of 'plot')

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
        num_cols : int, default 3
            Number of columns for the figure; the rows will be deduced
            accordingly
        metric_legend : bool, default True
            Whether the metric label should be displayed within the plot (as
            label of the y-axis)
        xtick_labelbottom : bool, default False
            If True, the label ticks (dates) in the xaxis will be displayed at
            each row. Otherwise they will only be displayed at the bottom row
        method : string, default 'bar'
            The matplotlib method to use for the plot. Options are: 'plot' or
            'bar'
        method_kws : dict
            Keyword arguments to be passed to the matplotlib plotting method
            as indicated in the `method` argument
        fmt : str, default '--o'
            A format string for the `plt.plot`. Ignored if `method` is 'bar'
        subplots_adjust_kws: dict, optional
            Keyword arguments to be passed to `plt.subplots_adjust`

        Returns
        -------
        fig, ax : tuple
            - figure object
            - axis object with the plot drawn onto it
        """
        return super(GradientAnalysis, self).plot_metrics(
            class_val=class_val, metrics=metrics, num_cols=num_cols,
            metric_legend=metric_legend, xtick_labelbottom=xtick_labelbottom,
            method=method, method_kws=method_kws, fmt=fmt,
            subplots_adjust_kws=subplots_adjust_kws)


class BufferAnalysis(GradientAnalysis):
    def __init__(self, landscape, base_mask, buffer_dists, base_mask_crs=None,
                 landscape_crs=None, landscape_transform=None, metrics=None,
                 classes=None, metrics_kws={}):

        # first check that we meet the package dependencies
        if not geo_imports:
            raise ImportError(
                "The `BufferAnalysis` class requires the geopandas package. "
                "For better performance, we strongly suggest that you install "
                "its cythonized version via conda-forge as in:\nconda install "
                "-c conda-forge/label/dev geopandas\n See "
                "https://github.com/geopandas/geopandas for more information "
                "about installing geopandas")

        # get `buffer_masks_arr` from a base geometry and a list of buffer
        # distances
        # 1. get a GeoSeries with the base mask geometry
        if isinstance(base_mask, BaseGeometry):
            if base_mask_crs is None:
                raise ValueError(
                    "If `base_mask` is a shapely geometry, `base_mask_crs` "
                    "must be provided")
            # BufferSpatioTemporalAnalysis.get_buffer_masks_gser(
            base_mask_gser = gpd.GeoSeries(base_mask, crs=base_mask_crs)
        else:
            # we assume that `base_mask` is a geopandas GeoSeries
            if base_mask.crs is None:
                if base_mask_crs is None:
                    raise ValueError(
                        "If `base_mask` is a naive geopandas GeoSeries (with "
                        "no crs set), `base_mask_crs` must be provided")
                else:
                    base_mask_gser = base_mask.copy(
                    )  # avoid alias/ref problems
                    base_mask_gser.crs = base_mask_crs
            else:
                base_mask_gser = base_mask

        # 2. get the crs, transform and shape of the landscapes
        if isinstance(landscape, Landscape):
            if landscape_crs is None:
                raise ValueError(
                    "If passing `Landscape` objects (instead of geotiff "
                    "filepaths), `landscapes_crs` must be provided")
            if landscape_transform is None:
                raise ValueError(
                    "If passing `Landscape` objects (instead of geotiff "
                    "filepaths), `landscapes_transform` must be provided")
            landscape_shape = landscape.landscape_arr.shape
        else:
            with rasterio.open(landscape) as src:
                landscape_crs = src.crs
                landscape_transform = src.transform
                landscape_shape = src.height, src.width

        # 3. buffer around base mask
        avg_longitude = base_mask_gser.to_crs({
            'init': 'epsg:4326'
        }).unary_union.centroid.x
        # trick from OSMnx to be able to buffer in meters
        utm_zone = int(np.floor((avg_longitude + 180) / 6.) + 1)
        utm_crs = {
            'datum': 'WGS84',
            'ellps': 'WGS84',
            'proj': 'utm',
            'zone': utm_zone,
            'units': 'm'
        }
        base_mask_geom = base_mask_gser.to_crs(utm_crs).iloc[0]
        base_masks_gser = gpd.GeoSeries([
            base_mask_geom.buffer(buffer_dist) for buffer_dist in buffer_dists
        ], index=buffer_dists, crs=utm_crs).to_crs(landscape_crs)

        # 4. rasterize each mask
        num_rows, num_cols = landscape_shape
        buffer_masks_arr = np.zeros((len(buffer_dists), num_cols, num_cols),
                                    dtype=np.uint8)
        for i in range(len(buffer_dists)):
            buffer_masks_arr[i] = features.rasterize(
                [base_masks_gser.iloc[i]], out_shape=landscape_shape,
                transform=landscape_transform, dtype=np.uint8)

        buffer_masks_arr = buffer_masks_arr.astype(bool)

        # now we can call the parent's init with the landscape and the
        # constructed buffer_masks_arr
        super(BufferAnalysis, self).__init__(
            landscape, buffer_masks_arr, 'buffer_dists', buffer_dists,
            metrics=metrics, classes=classes, metrics_kws=metrics_kws)
