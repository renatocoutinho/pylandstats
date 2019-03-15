import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pylandstats as pls

plt.switch_backend('agg')  # only for testing purposes


class TestLandscape(unittest.TestCase):
    def setUp(self):
        ls_arr = np.load('tests/input_data/ls250_06.npy')
        self.ls = pls.Landscape(ls_arr, res=(250, 250))

    def test_io(self):
        ls = pls.read_geotiff('tests/input_data/ls250_06.tif')
        # resolutions are not exactly 250, they are between [249, 251], so we
        # need to use a large delta
        self.assertAlmostEqual(ls.cell_width, 250, delta=1)
        self.assertAlmostEqual(ls.cell_height, 250, delta=1)
        self.assertAlmostEqual(ls.cell_area, 250 * 250, delta=250)

    def test_metrics_parameters(self):
        ls = self.ls

        for patch_metric in pls.Landscape.PATCH_METRICS:
            method = getattr(ls, patch_metric)
            self.assertIsInstance(method(), pd.DataFrame)
            self.assertIsInstance(method(class_val=ls.classes[0]), pd.Series)

        for class_metric in pls.Landscape.CLASS_METRICS:
            self.assertTrue(
                np.isreal(getattr(ls, class_metric)(class_val=ls.classes[0])))

        for landscape_metric in pls.Landscape.LANDSCAPE_METRICS:
            self.assertTrue(np.isreal(getattr(ls, landscape_metric)()))

    def test_metric_dataframes(self):
        ls = self.ls
        patch_df = ls.compute_patch_metrics_df()
        self.assertTrue(
            np.all(
                patch_df.columns.drop('class_val') ==
                pls.Landscape.PATCH_METRICS))
        self.assertEqual(patch_df.index.name, 'patch_id')
        self.assertRaises(ValueError, ls.compute_patch_metrics_df, ['foo'])

        class_df = ls.compute_class_metrics_df()
        self.assertEqual(
            len(class_df.columns.difference(pls.Landscape.CLASS_METRICS)), 0)
        self.assertEqual(class_df.index.name, 'class_val')
        self.assertRaises(ValueError, ls.compute_class_metrics_df, ['foo'])

        landscape_df = ls.compute_landscape_metrics_df()
        self.assertEqual(
            len(
                landscape_df.columns.difference(
                    pls.Landscape.LANDSCAPE_METRICS)), 0)
        self.assertEqual(len(landscape_df.index), 1)
        self.assertRaises(ValueError, ls.compute_landscape_metrics_df, ['foo'])

    def test_landscape_metrics_value_ranges(self):
        ls = self.ls

        # basic tests of the `Landscape` class' attributes
        self.assertNotIn(ls.nodata, ls.classes)
        self.assertGreater(ls.landscape_area, 0)

        class_val = ls.classes[0]
        # label_arr = ls._get_label_arr(class_val)

        # patch-level metrics
        assert (ls.area()['area'] > 0).all()
        assert (ls.perimeter()['perimeter'] > 0).all()
        assert (ls.perimeter_area_ratio()['perimeter_area_ratio'] > 0).all()
        assert (ls.shape_index()['shape_index'] >= 1).all()
        _fractal_dimension_ser = ls.fractal_dimension()['fractal_dimension']
        assert (_fractal_dimension_ser >= 1).all() and (_fractal_dimension_ser
                                                        <= 2).all()
        # TODO: assert 0 <= ls.contiguity_index(patch_arr) <= 1
        # TODO: assert 0 <= ls.euclidean_nearest_neighbor(patch_arr) <= 1
        # TODO: assert 0 <= ls.proximity(patch_arr) <= 1

        # class-level metrics
        assert ls.total_area(class_val) > 0
        assert 0 < ls.proportion_of_landscape(class_val) < 100
        assert ls.patch_density(class_val) > 0
        assert 0 < ls.largest_patch_index(class_val) < 100
        assert ls.total_edge(class_val) >= 0
        assert ls.edge_density(class_val) >= 0

        # the value ranges of mean, area-weighted mean and median aggregations
        # are going to be the same as their respective original metrics
        mean_suffixes = ['_mn', '_am', '_md']
        # the value ranges of the range, standard deviation and coefficient of
        # variation  will always be nonnegative as long as the means are
        # nonnegative as well (which is the case of all of the metrics
        # implemented so far)
        var_suffixes = ['_ra', '_sd', '_cv']

        for mean_suffix in mean_suffixes:
            assert getattr(ls, 'area' + mean_suffix)(class_val) > 0
            assert getattr(ls,
                           'perimeter_area_ratio' + mean_suffix)(class_val) > 0
            assert getattr(ls, 'shape_index' + mean_suffix)(class_val) >= 1
            assert 1 <= getattr(
                ls, 'fractal_dimension' + mean_suffix)(class_val) <= 2
            # assert 0 <= getattr(
            #     ls, 'contiguity_index' + mean_suffix)(class_val) <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)(class_val) >= 0
            # assert getattr(
            #     ls, 'euclidean_nearest_neighbor' + mean_suffix)(class_val) >

        for var_suffix in var_suffixes:
            assert getattr(ls, 'area' + mean_suffix)(class_val) >= 0
            assert getattr(ls,
                           'perimeter_area_ratio' + var_suffix)(class_val) >= 0
            assert getattr(ls, 'shape_index' + var_suffix)(class_val) >= 0
            assert getattr(ls,
                           'fractal_dimension' + var_suffix)(class_val) >= 0
            # assert getattr(
            #    ls, 'contiguity_index' + var_suffix)(class_val) >= 0
            # assert getattr(ls, 'proximity' + var_suffix)(class_val) >= 0
            # assert getattr(
            #     ls, 'euclidean_nearest_neighbor' + var_suffix)(
            #         class_val) >= 0

        # TODO: assert 0 < ls.interspersion_juxtaposition_index(
        #           class_val) <= 100
        assert ls.landscape_shape_index(class_val) >= 1

        # landscape-level metrics
        assert ls.total_area() > 0
        assert ls.patch_density() > 0
        assert 0 < ls.largest_patch_index() < 100
        assert ls.total_edge() >= 0
        assert ls.edge_density() >= 0
        assert 0 < ls.largest_patch_index() <= 100
        assert ls.total_edge() >= 0
        assert ls.edge_density() >= 0

        # for class_val in ls.classes:
        #     print('num_patches', class_val, ls._get_num_patches(class_val))
        #     print('patch_areas', len(ls._get_patch_areas(class_val)))

        # raise ValueError

        for mean_suffix in mean_suffixes:
            assert getattr(ls, 'area' + mean_suffix)() > 0
            assert getattr(ls, 'perimeter_area_ratio' + mean_suffix)() > 0
            assert getattr(ls, 'shape_index' + mean_suffix)() >= 1
            assert 1 <= getattr(ls, 'fractal_dimension' + mean_suffix)() <= 2
            # assert 0 <= getattr(ls, 'contiguity_index' + mean_suffix)() <= 1
            # assert getattr(ls, 'proximity' + mean_suffix)() >= 0
            # assert getattr(ls,
            #                'euclidean_nearest_neighbor' + mean_suffix)() > 0
        for var_suffix in var_suffixes:
            assert getattr(ls, 'area' + var_suffix)() > 0
            assert getattr(ls, 'perimeter_area_ratio' + var_suffix)() >= 0
            assert getattr(ls, 'shape_index' + var_suffix)() >= 0
            assert getattr(ls, 'fractal_dimension' + var_suffix)() >= 0
            # assert getattr(ls, 'contiguity_index' + var_suffix)() >= 0
            # assert getattr(ls, 'proximity' + var_suffix)() >= 0
            # assert getattr(ls,
            #                'euclidean_nearest_neighbor' + var_suffix)() >= 0

        # TODO: assert 0 < ls.contagion() <= 100
        # TODO: assert 0 < ls.interspersion_juxtaposition_index() <= 100
        assert ls.shannon_diversity_index() >= 0

    def test_plot_landscape(self):
        # returned axis must be instances of matplotlib axes
        self.assertIsInstance(self.ls.plot_landscape(), plt.Axes)


class TestSpatioTemporalAnalysis(unittest.TestCase):
    def setUp(self):
        self.landscapes = [
            pls.Landscape(np.load(fp), res=(250, 250)) for fp in
            ['tests/input_data/ls250_06.npy', 'tests/input_data/ls250_12.npy']
        ]
        self.landscape_fps = [
            'tests/input_data/ls250_06.tif', 'tests/input_data/ls250_12.tif'
        ]
        self.dates = [2006, 2012]
        self.inexistent_class_val = 999

    def test_spatiotemporalanalysis_init(self):
        for landscapes in (self.landscapes, self.landscape_fps):
            # test that constructing a SpatioTemporalAnalysis with inexistent
            # metrics and inexistent classes raises a ValueError
            self.assertRaises(ValueError, pls.SpatioTemporalAnalysis,
                              self.landscapes, metrics=['foo'])
            self.assertRaises(ValueError, pls.SpatioTemporalAnalysis,
                              self.landscape_fps, metrics=['foo'])
            self.assertRaises(ValueError, pls.SpatioTemporalAnalysis,
                              self.landscapes,
                              classes=[self.inexistent_class_val])
            self.assertRaises(ValueError, pls.SpatioTemporalAnalysis,
                              self.landscape_fps,
                              classes=[self.inexistent_class_val])

            # test that constructing a SpatioTemporalAnalysis with a `dates`
            # argument that mismatch the temporal snapshots defined in
            # `landscapes` raises a ValueError
            self.assertRaises(ValueError, pls.SpatioTemporalAnalysis,
                              self.landscapes, dates=[2012])
            self.assertRaises(ValueError, pls.SpatioTemporalAnalysis,
                              self.landscape_fps, dates=[2012])

    def test_spatiotemporalanalysis_dataframes(self):
        # test with the default constructor
        sta = pls.SpatioTemporalAnalysis(self.landscapes)

        # test that `class_metrics_df` and `landscape_metrics_df` are well
        # constructed
        class_metrics_df = sta.class_metrics_df
        self.assertTrue(
            np.all(class_metrics_df.columns == pls.Landscape.CLASS_METRICS))
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [sta.classes, sta.dates])))
        landscape_metrics_df = sta.landscape_metrics_df
        self.assertTrue(
            np.all(landscape_metrics_df.columns ==
                   pls.Landscape.LANDSCAPE_METRICS))
        self.assertTrue(np.all(landscape_metrics_df.index == sta.dates))

        # now test the same but with an analysis that only considers a subset
        # of metrics and a subset of classes
        sta_metrics = ['total_area', 'edge_density', 'proportion_of_landscape']
        sta_classes = self.landscapes[0].classes[:2]
        sta = pls.SpatioTemporalAnalysis(self.landscapes, metrics=sta_metrics,
                                         classes=sta_classes, dates=self.dates)

        class_metrics_df = sta.class_metrics_df
        self.assertTrue(
            np.all(class_metrics_df.columns == np.intersect1d(
                sta_metrics, pls.Landscape.CLASS_METRICS)))
        self.assertTrue(
            np.all(class_metrics_df.index == pd.MultiIndex.from_product(
                [sta_classes, self.dates])))
        landscape_metrics_df = sta.landscape_metrics_df
        self.assertTrue(
            np.all(landscape_metrics_df.columns == np.intersect1d(
                sta_metrics, pls.Landscape.LANDSCAPE_METRICS)))
        self.assertTrue(np.all(landscape_metrics_df.index == self.dates))

    def test_spatiotemporalanalysis_metric_kws(self):
        # Instantiate two spatiotemporal analysises, one with FRAGSTATS'
        # defaults and the other with keyword arguments specifying the total
        # area in meters and including the boundary in the computation of the
        # total edge.
        sta = pls.SpatioTemporalAnalysis(self.landscapes)
        sta_kws = pls.SpatioTemporalAnalysis(
            self.landscapes, metrics_kws={
                'total_area': {
                    'hectares': False
                },
                'total_edge': {
                    'count_boundary': True
                }
            })

        # For all dates and all classes, metric values in hectares should be
        # less than in meters, and excluding boundaries should be less or
        # equal than including them
        for date in sta.dates:
            landscape_metrics = sta.landscape_metrics_df.loc[date]
            landscape_metrics_kws = sta_kws.landscape_metrics_df.loc[date]
            self.assertLess(landscape_metrics['total_area'],
                            landscape_metrics_kws['total_area'])
            self.assertLessEqual(landscape_metrics['total_edge'],
                                 landscape_metrics_kws['total_edge'])

            for class_val in sta.classes:
                class_metrics = sta.class_metrics_df.loc[class_val, date]
                class_metrics_kws = sta_kws.class_metrics_df.loc[class_val,
                                                                 date]

                # It could be that for some dates, some classes are not
                # present within the landscape snapshot. In such case, all of
                # the metrics will be `nan`, both for the analysis with and
                # without keyword arguments. Otherwise, we just perform the
                # usual checks
                if class_metrics.isnull().all():
                    self.assertTrue(class_metrics_kws.isnull().all())
                else:
                    self.assertLess(class_metrics['total_area'],
                                    class_metrics_kws['total_area'])
                    self.assertLessEqual(class_metrics['total_edge'],
                                         class_metrics_kws['total_edge'])

    def test_spatiotemporalanalysis_plot_metrics(self):
        sta = pls.SpatioTemporalAnalysis(self.landscapes, dates=self.dates)

        existent_class_val = sta.classes[0]

        # inexistent metrics should raise a ValueError
        self.assertRaises(ValueError, sta.plot_metric, 'foo')
        # inexistent classes should raise a ValueError
        self.assertRaises(ValueError, sta.plot_metric, 'patch_density',
                          {'class_val': self.inexistent_class_val})
        # `proportion_of_landscape` can only be computed at the class level,
        # so plotting it at the landscape level (with the default argument
        # `class_val=None`) must raise a ValueError
        self.assertRaises(ValueError, sta.plot_metric,
                          'proportion_of_landscape')
        # conversely, `shannon_diversity_index` can only be computed at the
        # landscape level, so plotting it at the class level must raise a
        # ValueError
        self.assertRaises(ValueError, sta.plot_metric,
                          'shannon_diversity_index',
                          {'class_val': existent_class_val})

        # TODO: test legend and figsize

        ax = sta.plot_metric('patch_density', class_val=None)
        self.assertEqual(len(ax.lines), 1)
        ax = sta.plot_metric('patch_density', class_val=existent_class_val,
                             ax=ax)
        self.assertEqual(len(ax.lines), 2)

        fig, axes = sta.plot_metrics(class_val=existent_class_val,
                                     metrics=['edge_density', 'patch_density'])
        self.assertEqual(len(axes), 2)

    def test_plot_landscape(self):
        sta = pls.SpatioTemporalAnalysis(self.landscapes)

        fig, axes = sta.plot_landscapes()

        # there must be one column for each landscape
        self.assertEqual(len(axes), len(sta))

        # returned axes must be instances of matplotlib axes
        for ax in axes:
            self.assertIsInstance(ax, plt.Axes)


class TestMultiSpatioTemporalAnalysis(unittest.TestCase):
    def setUp(self):
        self.landscape_list_dict = {
            res: [
                pls.Landscape(np.load(fp), res=(res, res)) for fp in [
                    'tests/input_data/ls{}_06.npy'.format(res),
                    'tests/input_data/ls{}_12.npy'.format(res)
                ]
            ]
            for res in [100, 250]
        }
        self.landscape_fp_dict = {
            res: [
                'tests/input_data/ls{}_06.tif'.format(res),
                'tests/input_data/ls{}_12.tif'.format(res)
            ]
            for res in [100, 250]
        }
        self.dates = [2006, 2012]
        self.inexistent_class_val = 999

    def test_multispatiotemporalanalysis_init(self):
        # test constructing a `MultiSpatioTemporalAnalysis` from a dict that
        # maps each factor to a list of `Landscape` instances or to a list of
        # filepaths
        for landscape_dict in [
                self.landscape_list_dict, self.landscape_fp_dict
        ]:
            # construct an instance without arguments
            msta = pls.MultiSpatioTemporalAnalysis(landscape_dict)
            first_sta = next(iter(msta.landscape_dict.values()))
            # test that the dict values are `SpatioTemporalAnalysis`
            self.assertIsInstance(first_sta, pls.SpatioTemporalAnalysis)

            # test that the arguments are passed to
            # `SpatioTemporalAnalysis.__init__` and that they raise the
            # corresponding errors
            self.assertRaises(ValueError, pls.MultiSpatioTemporalAnalysis,
                              landscape_dict, dates=[2012])
            self.assertRaises(ValueError, pls.MultiSpatioTemporalAnalysis,
                              landscape_dict, metrics=['foo'])
            self.assertRaises(ValueError, pls.MultiSpatioTemporalAnalysis,
                              landscape_dict,
                              classes=[self.inexistent_class_val])

        # note that we do not test constructing a `MultiSpatioTemporalAnalysis`
        # from a dict that maps each factor to `SpatioTemporalAnalysis`
        # instances, since in such case, `MultiSpatioTemporalAnalysis.__init__`
        # ignores the rest of arguments

    def test_multispatiotemporal_plot_metric(self):
        # test plotting a `MultiSpatioTemporalAnalysis` from a dict that maps
        # each factor to a list of `Landscape` instances or to a list of
        # filepaths
        for landscape_dict in [
                self.landscape_list_dict, self.landscape_fp_dict
        ]:
            msta = pls.MultiSpatioTemporalAnalysis(landscape_dict)
            first_sta = next(iter(msta.landscape_dict.values()))
            # plot a 'random' metric
            ax = msta.plot_metric('patch_density')
            # test that the plotted line of each factor has a data point for
            # each landscape of the `SpatioTemporalAnalysis`
            for line in ax.lines:
                self.assertEqual(len(line.get_xdata()), len(first_sta))

        # now test constructing a `MultiSpatioTemporalAnalysis` from a dict
        # that maps each factor to `SpatioTemporalAnalysis` instances. Note
        # that in this case, `MultiSpatioTemporalAnalysis.__init__` will
        # ignore the rest of arguments, and it is the user's responsability to
        # correctly construct each of the `SpatioTemporalAnalysis` instances.
        # In this test, we will construct a dummy `SpatioTemporalAnalysis`
        # with only one snapshot for the 100 resolution, and another
        # `SpatioTemporalAnalysis` with two snapshots for the 250 resolution.
        landscape_sta_dict = {
            100:
            pls.SpatioTemporalAnalysis(['tests/input_data/ls100_06.tif']),
            250:
            pls.SpatioTemporalAnalysis([
                'tests/input_data/ls250_06.tif',
                'tests/input_data/ls250_12.tif'
            ])
        }
        msta = pls.MultiSpatioTemporalAnalysis(landscape_sta_dict)
        ax = msta.plot_metric('patch_density')
        # test that the plotted line of each factor has a data point for
        # each landscape of ITS RESPECTIVE `SpatioTemporalAnalysis` - note
        # that in this case, the first `SpatioTemporalAnalysis` will have a
        # `len` of 1, whereas the second will have a `len` of 2.
        for sta, line in zip(msta.landscape_dict.values(), ax.lines):
            self.assertEqual(len(line.get_xdata()), len(sta))

        # test that we do not have labels with `factor_label=False`
        self.assertTrue(ax.get_legend_handles_labels()[1])
        ax = msta.plot_metric('patch_density', factor_legend=False)
        self.assertFalse(ax.get_legend_handles_labels()[1])

        # very basic test for plot metrics
        fig, axes = msta.plot_metrics(
            metrics=['edge_density', 'patch_density'])
        self.assertEqual(len(axes), 2)
        for ax in axes:
            for sta, line in zip(msta.landscape_dict.values(), ax.lines):
                self.assertEqual(len(line.get_xdata()), len(sta))
