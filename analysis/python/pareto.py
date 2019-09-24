#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

import numpy as np


PARETO_DOMINATED = -2
BELOW_CURVE = -1
INCOMPARABLE = 0
ABOVE_CURVE = 1
PARETO_DOMINANT = 2


def slope_between_pts(left_point, right_point):
    x1, y1 = left_point
    x2, y2 = right_point

    if x2 == x1:
        raise ZeroDivisionError("got x1 = x2 = {}".format(x1))

    # # denom = (x2 - x1) + 1e-20
    # denom = (x2 - x1)

    # if denom == 0:
    #     raise ZeroDivisionError("got

    # return (y2 - y1) / denom

    return (y2 - y1) / (x2 - x1)


def linear_interp_at_x_position(x, left_point, right_point):
    x1, y1 = left_point
    slope = slope_between_pts(left_point, right_point)
    return y1 + (slope * (x - x1))

    # x2, y2 = right_point

    # print("linear interping between points: ", left_point, right_point)

    # if x2 == x1:
    #     # raise ZeroDivisionError("got x1 = x2 = {}".format(x1))
    # denom = (x2 - x1) + 1e-20
    # slope = (y2 - y1) / denom
    # y = y1 + (slope * (x - x1))

    # assert x1 <= x <= x2
    # assert (min(y1, y2) - 1e-7) <= y <= (max(y1, y2) + 1e-7)

    # return y


def cmp_to_linear_interp_curve(a, b_points_sorted, only_check_pareto=False):
    # assumes up and to the right is better
    points = np.atleast_2d(b_points_sorted)
    a = np.array(a)
    assert a.ndim == 1
    assert len(a) == 2
    assert points.ndim == 2

    xvals = points[:, 0]
    # yvals = points[:, 1]
    x_a, y_a = a

    # # handle case of x_a in x_vals
    # eq_idxs = np.where(x_a == xvals)[0]
    # if len(eq_idxs):
    #     # print("x in xvals! ")
    #     eq_yvals = yvals[eq_idxs]
    #     if np.all(y_a < np.min(eq_yvals)):
    #         return PARETO_DOMINATED  # worse than all b vals at this idx
    #     if np.all(y_a > np.max(eq_yvals)):
    #         return PARETO_DOMINANT  # better than all b vals at this idx
    #     return INCOMPARABLE  # in between b vals at this x idx

    # check pareto dominance; note that we check for being dominated before
    # being dominant so that won't return dominant unless actually on pareto
    # frontier
    le_mask = (a <= points).astype(np.int8)
    all_le = le_mask.sum(axis=1) == points.shape[1]
    worse_mask = all_le & ((a < points).sum(axis=1, keepdims=True) > 0)
    if np.any(worse_mask):
        return PARETO_DOMINATED
    ge_mask = (a >= points).astype(np.int8)
    all_ge = ge_mask.sum(axis=1) == points.shape[1]
    better_mask = all_ge & ((a > points).sum(axis=1, keepdims=True) > 0)
    if np.any(better_mask):
        return PARETO_DOMINANT

    # if x_a in xvals:  # if equal to some xval but not pareto improvement
    #     return INCOMPARABLE

    # print("did pareto check!")

    # handle case of x_a past endpoints of xvals
    if x_a < xvals[0] or x_a > xvals[-1]:
        return INCOMPARABLE
    # if x_a < xvals[0]:
    #     worse = y_a > np.max(yvals)
    #     return PARETO_DOMINATED if worse else INCOMPARABLE

    if only_check_pareto:
        return INCOMPARABLE

    # if we got to here, min(xvals) < x_a < max(xvals) and no dominance
    idx = np.sum(x_a > xvals)
    left_idx, right_idx = idx - 1, idx

    # no pareto dominance; check if above/below curve
    left_pt, right_pt = points[left_idx], points[right_idx]
    y_hat = linear_interp_at_x_position(x_a, left_pt, right_pt)

    # print("did linear interp!")

    # print("np.sum(x_a > xvals)", np.sum(x_a > xvals))
    # print("idx: ", idx)
    # print("left and right pts: ", left_pt, right_pt)
    # print("y, yhat = ", y_a, y_hat)

    if y_a == y_hat:
        return INCOMPARABLE
    return ABOVE_CURVE if y_a > y_hat else BELOW_CURVE


def pareto_cmp_to_points(a, b_points_sorted):
    return cmp_to_linear_interp_curve(
        a, b_points_sorted, only_check_pareto=True)


def extract_curve_vertices(points, how='convex_hull'):
    # remove duplicate points and sort in ascending order of x vertices
    points = np.atleast_2d(points)
    assert points.ndim == 2

    if how is None:
        return points

    points = np.unique(points, axis=0)

    if how == 'dedup':  # just deduplicate
        return points
    if len(points) == 1:
        return points

    yvals = points[:, 1]
    yvals_are_decreasing = np.all(np.diff(yvals) < 0)
    use_pareto_frontier = ('pareto' in how) or yvals_are_decreasing

    # extract points on pareto frontier
    if use_pareto_frontier:
        # keep_idxs = set(np.arange(len(points)))
        keep_idxs = []
        other_points = points.copy()
        # max_yval = np.max(points[:, 1]) + 1
        for idx, row in enumerate(points):
            # other_points = points[keep_idxs - set([idx])]
            # xval = points[idx][0]
            # other_points[idx] = (xval, max_yval)
            other_points = np.vstack([points[:idx], points[idx + 1:]])

            # print("idx = {}; other points = ".format(idx))
            # print(other_points)

            result = pareto_cmp_to_points(row, other_points)
            if result != PARETO_DOMINATED:
                keep_idxs.append(idx)

            # print("----\nidx = {}; result = {}; row = {}".format(idx, result, row))
            # print("other points = ")
            # print(other_points)

            # other_points[idx] = points[idx]

        points = points[keep_idxs]
        if how == 'pareto':
            return points

    # this is the harder case; need to extract convex hull of points, defined
    # as set of points that are greater than any convex combo of other points
    # (this assumes higher is better). We do this by enforcing the invariant
    # that the slope between successive points has to be nonincreasing as you
    # traverse them from left to right (in terms of x coordinate)
    assert how in ('convex_hull', 'pareto_hull')  # true if we got to here

    # if how == 'convex_hull' and len(points) > 2:
    if how == 'convex_hull':
        # didn't rm pareto dominated points, so might have multiple y vals for
        # one x val; this will yield slopes of infinity
        new_points = []
        xvals = points[:, 0]
        for xval in np.unique(xvals):
            rows = np.where(xvals == xval)[0]
            max_y = np.max(points[rows, 1])
            new_points.append((xval, max_y))

        # print("new points: ", new_points)
        points = np.array(new_points)

    new_points = points
    # keep_idxs = [0, N - 1]
    need_another_pass = len(points) > 2
    while need_another_pass:
        N = len(new_points)
        any_point_removed = False
        for idx in np.arange(1, N - 1):  # always keep first and last
            # left_idx, right_idx = idx - 1, idx + 1
            # left_pt, right_pt = points[left_idx], points[right_idx]
            # this_pt = points[idx]
            left_slope = slope_between_pts(points[idx - 1], points[idx])
            right_slope = slope_between_pts(points[idx + 1], points[idx])
            if right_slope > left_slope:
                any_point_removed = True
                new_points = np.vstack([new_points[:idx], new_points[idx + 1:]])
                break

        need_another_pass = any_point_removed and len(new_points) > 2

    return new_points


def compare_curves(points_a, points_b,
                   x_higher_better=True, y_higher_better=True,
                   # curve_algo='convex_hull'):
                   a_curve_algo=None, b_curve_algo='pareto_hull',
                   extrapolate_b=False):
    # map each (x, y) tuple a in points_a to
    #   -2 if a is pareto dominated by at least one point in points_b
    #   -1 if a is under the linearly interpolated curve of points_b
    #   1 if a is above the linearly interpolated curve of points_b
    #   2 if a is above the linearly interpolated curve of points_b and pareto
    #       dominates at least one point in points_b
    #   0 otherwise
    #
    # returns an int vector with same length as points_a
    #
    # if curve_algo == 'convex_hull', will throw away elements of points_b
    # that aren't greater than all convex combinations of remaining points
    # evaluated at their first coordinate
    # if curve_algo == 'pareto', will throw away elements of points_b
    # that aren't pareto dominated by other points in bs
    # if curve_algo == 'all', won't throw away any elements of points_b
    # except duplicates (which doesn't affect the output, I think?)
    #
    # and note that the statements about above/below or greater/less are
    # reversed in accordance with x_higher_better and y_higher_better
    #
    # also note that we return results for sorted, deduplicated versions
    # of points_a and points_b after taking into account x_higher_better and
    # y_higher_better; so basically you just shouldn't assume there's any
    # correspondence between the original elements of points_a and points_b
    # and the elements of the returned array

    points_a = np.asarray(points_a)
    points_b = np.asarray(points_b)

    if points_a.ndim == 1:
        assert len(points_a) == 2
        points_a = np.atleast_2d(points_a)
    if points_b.ndim == 1:
        assert len(points_b) == 2
        points_b = np.atleast_2d(points_b)

    if not x_higher_better:
        points_a[:, 0] *= -1
        points_b[:, 0] *= -1
    if not y_higher_better:
        points_a[:, 1] *= -1
        points_b[:, 1] *= -1

    points_a = np.unique(points_a, axis=0)
    points_b = np.unique(points_b, axis=0)

    points_a = extract_curve_vertices(points_a, how=a_curve_algo)
    curve_points_b = extract_curve_vertices(points_b, how=b_curve_algo)

    ret = np.empty(len(points_a), dtype=np.int8)
    for i, pt in enumerate(points_a):
        # print("i, pt = ", i, pt)
        ret[i] = cmp_to_linear_interp_curve(pt, curve_points_b)

    # if willing to assume that curve is a concave function of x, linearly
    # extrapolating from the first/last two elements of points_b provides
    # an optimistic estimate of how good its y values could be at the xvals
    # from points_a; if a still beats these, count it as beating b
    if extrapolate_b and len(points_b) > 1:
        min_x = np.min(points_a[:, 0])
        max_x = np.max(points_a[:, 0])
        y_start = linear_interp_at_x_position(min_x, points_b[0], points_b[1])
        y_end = linear_interp_at_x_position(max_x, points_b[-2], points_b[-1])
        point_start = np.atleast_2d([min_x, y_start])
        point_end = np.atleast_2d([max_x, y_end])
        points_b = np.vstack([point_start, points_b, point_end])

        # print("augmented points_b:")
        # print(points_b)

        ret2 = compare_curves(points_a, points_b,
                              a_curve_algo=None, b_curve_algo=None)
        # don't allow pareto dominance from extrapolated points; this both
        # seems like too strong a conclusion from extrapolated data, and also
        # makes eval sensitive to whether we evaluate extrapolations at
        # *exactly* the same x positions as elements of points_a
        ret2 = np.minimum(ret2, ABOVE_CURVE)
        ret = np.maximum(ret, ret2)  # only let linear extrapolation help

    return ret

    # # return ret_a

    # # if not use_reverse_order_info:
    # #     return ret_a

    # # this is basically to handle case of a being above b, but without x vals
    # # that straddle those of b; we can see that the curve through the a points
    # # is above the curve through the b points, but when you compare each point
    # # in a on its own, the endpoints will look like they're past the end of b
    # # and so not comparable
    # curve_points_a = extract_curve_vertices(points_a, how=curve_algo)
    # ret_b = np.empty(len(points_b), dtype=np.int8)
    # for i, pt in enumerate(points_b):
    #     x_b = pt[0]
    #     cmp_result = ret_b[i]



    #     # SELF: pick up here:



    #     a_left =
    #     a_right =


    #     # ret_b[i] = cmp_to_linear_interp_curve(pt, curve_points_a)

    # # ret = []
    # # for code_a, code_b in zip(ret_a, ret_b):
    # #     if code_a == INCOMPARABLE and code_b == BELOW_CURVE:
    # #         ret.append(ABOVE_CURVE)
    # #     else:
    # #         ret.append(code_a)

    return ret
