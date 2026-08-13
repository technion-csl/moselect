#!/usr/bin/env python3
"""CSV-backed persistent state for the MosSelect adaptive layout generator.

Each class here wraps a small pandas DataFrame that is transparently
synced to a CSV file under the experiment directory, so that
`layout_generator.py` can resume an interrupted MosSelect run (each
`createLayouts.py` invocation only produces one layout, then exits; `make`
re-invokes it once measurements for the previous layout are available).

- `SubgroupsLog` (``subgroups.log``): one row per anchor layout produced by
  the static bootstrap phase (`LayoutGenerator.createInitialLayoutsStatically`),
  tracking each subgroup's (i.e., the interval between two adjacent anchors)
  total/remaining measurement budget.
- `StateLog` (``<right>_<left>_state.log`` + shared ``layout_pages.log``):
  one row per layout generated while scanning a specific subgroup (bounded
  by a ``right`` and ``left`` anchor layout), recording how each layout was
  derived (scan direction/order/value, base layout, coverage predictions)
  plus a shared table of each layout's actual page list, enabling gap
  queries such as `getMaxGap`/`getNextIncrementBase`.
"""
import os
import pandas as pd
from ast import literal_eval
import os.path

class Singleton(type):
    """Metaclass ensuring a single shared instance per class (used by `SubgroupsLog`)."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args,
                                                                 **kwargs)
        return cls._instances[cls]


class Log():
    """Base class for a CSV-backed DataFrame log, keyed by layout name.

    Subclasses (`SubgroupsLog`, `StateLog`) provide `default_columns` and
    any extra construction logic; this base class provides generic
    load/save, per-layout field lookup, and real-coverage computation
    shared by both.

    Attributes:
        exp_dir: Root experiment directory.
        results_df: DataFrame of all measured layouts (``layout``,
            ``walk_cycles``, ...), used by `writeRealCoverage`.
        log_file: Full path to this log's CSV file.
        max_gap: Maximum acceptable real-coverage gap (percentage points).
        max_budget: Total layout budget for the experiment.
        default_columns: Column names for a freshly initialized (empty) log.
        df: The in-memory DataFrame backing this log.
        dry_run: If True, `writeLog`/`clear` are no-ops (debug mode).
    """

    def __init__(self,
                 exp_dir, results_df, log_name,
                 max_gap, max_budget, dry_run,
                 default_columns, converters=None):
        self.exp_dir = exp_dir
        self.results_df = results_df
        self.log_file = self.exp_dir + '/' + log_name
        self.max_gap = max_gap
        self.max_budget = max_budget
        self.default_columns = default_columns
        self.df = self.readLog(converters)
        self.dry_run = dry_run

    def readLog(self, converters=None):
        """Loads `self.df` from `log_file`, or initializes an empty DataFrame if it doesn't exist yet.

        Args:
            converters: Optional per-column parsing functions forwarded to
                `pandas.read_csv` (e.g., to parse stringified lists).

        Returns:
            pandas.DataFrame: The loaded (or freshly initialized) DataFrame.
        """
        if not os.path.isfile(self.log_file):
            self.df = pd.DataFrame(columns=self.default_columns)
        else:
            self.df = pd.read_csv(self.log_file, converters=converters)
        return self.df

    def writeLog(self):
        """Persists `self.df` to `log_file`, unless `dry_run` is set."""
        if not self.dry_run:
            self.df.to_csv(self.log_file, index=False)

    def clear(self):
        """Resets `self.df` to an empty DataFrame (does not delete the file), unless `dry_run` is set."""
        if not self.dry_run:
            self.df = pd.DataFrame(columns=self.default_columns)

    def empty(self):
        """Returns True if the log has no rows."""
        return self.df.empty

    def getField(self, layout_name, field_name):
        """Looks up a single field value for a given layout.

        Args:
            layout_name: Name of the layout to look up.
            field_name: Column name to read.

        Returns:
            The field's value, or ``None`` if `layout_name` is not present.
        """
        field_val = self.df.loc[self.df['layout'] == layout_name, field_name]
        field_val = field_val.to_list()
        if field_val == []:
            return None
        else:
            return field_val[0]

    def layoutExist(self, layout):
        """Returns True if `layout` already has a row in this log."""
        return len(self.df.query(f'layout == "{layout}"')) > 0

    def getRealCoverage(self, layout):
        """Returns `layout`'s measured real-coverage percentage (or None if unknown/unmeasured)."""
        return self.getField(layout, 'real_coverage')

    def getExpectedRealCoverage(self, layout):
        """Returns the real-coverage percentage that was predicted for `layout` when it was created."""
        return self.getField(layout, 'expected_real_coverage')

    def getPebsCoverage(self, layout):
        """Returns `layout`'s PEBS-predicted TLB-coverage percentage."""
        return self.getField(layout, 'pebs_coverage')

    def getLastRecord(self):
        """Returns the last row whose ``scan_base`` is not ``'other'`` (i.e., the last layout created by the normal scan flow), or None if there are none."""
        if self.empty():
            return None

        df = self.df.query('scan_base != "other"')
        if len(df) == 0:
            return None
        return df.iloc[-1]

    def getLastLayoutName(self):
        """
        Returns
        -------
        string
            returns the name of the last layout in the state log.
        """
        last_record = self.getLastRecord()
        assert last_record is not None,'getLastLayoutName: there is no state records'
        return last_record['layout']

    def getRecord(self, key_name, key_value):
        """Returns the first row where `key_name` equals `key_value`, or None."""
        record = self.df.query('{key} == "{value}"'.format(
            key=key_name,
            value=key_value))
        if record.empty:
            return None
        else:
            return record.iloc[0]

    def writeRealCoverage(self):
        """Computes and fills in `real_coverage` for any rows still marked as unmeasured (-1).

        For each such row, looks up the layout's `walk_cycles` in
        `results_df` (if available) and converts it into a real-coverage
        percentage: 0% at the maximum observed `walk_cycles`
        (slowest/no-hugepages) and 100% at the minimum (fastest/all-hugepages).
        Persists the updated log via `writeLog`.
        """
        max_walk_cycles = self.results_df['walk_cycles'].max()
        min_walk_cycles = self.results_df['walk_cycles'].min()
        delta_walk_cycles = max_walk_cycles - min_walk_cycles
        self.df['real_coverage'] = self.df['real_coverage'].astype(float)
        query = self.df.query('real_coverage == (-1)')
        for index, row in query.iterrows():
            layout = row['layout']
            walk_cycles = self.results_df.loc[self.results_df['layout'] == layout, 'walk_cycles'].iloc[0]
            real_coverage = (max_walk_cycles - walk_cycles) / delta_walk_cycles
            real_coverage *= 100
            self.df.loc[self.df['layout'] == layout, 'real_coverage'] = real_coverage
            self.df.loc[self.df['layout'] == layout, 'walk_cycles'] = walk_cycles
        self.writeLog()


class SubgroupsLog(Log, metaclass=Singleton):
    """Tracks the 9 static anchor layouts and each subgroup's measurement budget.

    A subgroup is the interval between two adjacent anchor layouts (sorted
    by real coverage); its budget is the number of additional layouts the
    adaptive phase (`LayoutGenerator.createNextLayoutDynamically`) is
    allowed to generate to close that interval's real-coverage gaps below
    `max_gap`. Larger initial gaps receive proportionally larger budgets
    (see `calculateBudget`). A `Singleton` since only one such log exists
    per experiment/process.
    """
    def __init__(self, exp_dir, results_df, max_gap, max_budget, dry_run):
        default_columns = [
            'layout', 'total_budget', 'remaining_budget',
            'pebs_coverage', 'real_coverage', 'walk_cycles']
        super().__init__(exp_dir, results_df, 'subgroups.log', max_gap, max_budget, dry_run, default_columns)

    def addRecord(self,
                  layout, pebs_coverage, writeLog=False):
        """Appends a new anchor layout row (budget/real-coverage left uninitialized as -1).

        Args:
            layout: Name of the new anchor layout.
            pebs_coverage: The anchor's PEBS-predicted coverage.
            writeLog: If True, immediately persists the log to disk.
        """
        new_row = pd.Series({
            'layout': layout,
            'total_budget': -1,
            'remaining_budget': -1,
            'pebs_coverage': pebs_coverage,
            'real_coverage': -1,
            'walk_cycles': -1 })
        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        if writeLog:
            self.writeLog()

    def getSubgroup(self, num):
        """Returns the (right, left) anchor-layout row pair bounding subgroup `num`.

        Assumes `self.df` is sorted by real coverage ascending (see
        `sortByRealCoverage`), so ``df.iloc[num]`` is the lower-coverage
        ("right") anchor and ``df.iloc[num+1]`` is the next, higher-coverage
        ("left") anchor.

        Args:
            num: Zero-based subgroup index (0..7 for the 9 static anchors).

        Returns:
            tuple[pandas.Series, pandas.Series]: ``(right, left)`` anchor
            rows.
        """
        right = self.df.iloc[num]
        left = self.df.iloc[num+1]
        return right, left

    def sortByRealCoverage(self):
        """Sorts `self.df` by measured `walk_cycles` descending (equivalent to real coverage ascending)."""
        #self.df = self.df.sort_values('real_coverage', ascending=True)
        self.df = self.df.sort_values('walk_cycles', ascending=False)

    def getExtraBudget(self):
        """Returns unallocated budget: `max_budget` minus (already-allocated total budget + anchor count)."""
        return self.max_budget - (self.getTotalBudget() + len(self.df))

    def calculateBudget(self):
        """Allocates each subgroup's measurement budget proportionally to its real-coverage gap.

        Must be called only after every anchor's `real_coverage` has been
        computed (via `Log.writeRealCoverage`). Sorts anchors by real
        coverage, computes the gap (``delta``) between each adjacent pair,
        and distributes the total available budget (``max_budget -
        len(anchors)``) across subgroups whose gap exceeds `max_gap`,
        proportionally to gap size (with a floor of ``delta / 3.5``
        layouts). Any rounding remainder is added to the last subgroup. A
        no-op if budgets were already calculated (``total_budget >= 0``).

        Raises:
            AssertionError: If any anchor's `real_coverage` is still
                unmeasured (-1).
        """
        query = self.df.query('real_coverage == (-1)')
        assert len(query) == 0, 'SubgroupsLog.calculateBudget was called before updating the subgroups real_coverage.'
        query = self.df.query('total_budget < 0')
        if len(query) == 0:
            return
        # sort the group layouts by walk-cycles/real_coverage
        self.sortByRealCoverage()
        # calculate the diff between each two adjacent layouts
        # (call it delta[i] for the diff between group[i] and group[i+1])
        self.df['delta'] = self.df['real_coverage'].diff().abs()
        self.df['delta'] = self.df['delta'].fillna(0)
        total_deltas = self.df.query(f'delta > {self.max_gap}')['delta'].sum()
        # budgest = 50-9: num_layouts(50) - subgroups_layouts(9)
        total_budgets = self.max_budget - len(self.df)
        for index, row in self.df.iterrows():
            delta = row['delta']
            # for each delta < self.max_gap assign budget=0
            if delta <= self.max_gap:
                budget = 0
            else:
                budget = int((delta / total_deltas) * total_budgets)
                budget = max(budget, int(delta / 3.5))
            self.df.at[index, 'total_budget'] = budget
            self.df.at[index, 'remaining_budget'] = budget
        # fix total budgets due to rounding
        rounded_total_budgets = self.df['total_budget'].sum()
        delta_budget = total_budgets - rounded_total_budgets
        self.df.at[index, 'total_budget'] = budget + delta_budget
        self.df.at[index, 'remaining_budget'] = budget + delta_budget

        self.writeLog()

    def decreaseRemainingBudget(self, layout):
        """Decrements `layout`'s subgroup's `remaining_budget` by one and persists the log."""
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = self.df.loc[self.df['layout'] == layout, 'remaining_budget']-1
        self.writeLog()

    def zeroAllBudgets(self):
        """Zeroes out every subgroup's remaining budget (see `zeroBudget`).

        Returns:
            int: The sum of all remaining budgets that were reclaimed.
        """
        remaining = 0
        for index, row in self.df.iterrows():
            layout = row['layout']
            remaining += self.zeroBudget(layout)
        return remaining

    def zeroBudget(self, layout):
        """Reclaims `layout`'s subgroup's remaining budget: sets `remaining_budget` to 0 and shrinks `total_budget` accordingly.

        Args:
            layout: The anchor layout identifying the subgroup.

        Returns:
            int: The remaining budget that was reclaimed.
        """
        total = self.getField(layout, 'total_budget')
        remaining = self.getField(layout, 'remaining_budget')
        self.df.loc[self.df['layout'] == layout, 'total_budget'] = total - remaining
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = 0
        self.writeLog()
        return remaining

    def addExtraBudget(self, layout, extra_budget):
        """Adds `extra_budget` to both the total and remaining budget of `layout`'s subgroup (e.g., budget reclaimed via `zeroAllBudgets`)."""
        self.df.loc[self.df['layout'] == layout, 'remaining_budget'] = self.df.loc[self.df['layout'] == layout, 'remaining_budget']+extra_budget
        self.df.loc[self.df['layout'] == layout, 'total_budget'] = self.df.loc[self.df['layout'] == layout, 'total_budget']+extra_budget
        self.writeLog()

    def getRightmostLayout(self):
        """Returns the anchor row with the highest `walk_cycles` (lowest real coverage, i.e. no-hugepages end)."""
        self.writeRealCoverage()
        df = self.df.sort_values('walk_cycles', ascending=False)
        return df.iloc[0]

    def getLeftmostLayout(self):
        """Returns the anchor row with the lowest `walk_cycles` (highest real coverage, i.e. all-hugepages end)."""
        self.writeRealCoverage()
        df = self.df.sort_values('walk_cycles', ascending=True)
        return df.iloc[0]

    def getRemainingBudget(self, left_layout):
        """Returns the remaining measurement budget for the subgroup identified by `left_layout`."""
        return self.getField(left_layout, 'remaining_budget')

    def getTotalRemainingBudget(self):
        """Returns the sum of remaining budgets across all subgroups."""
        return self.df['remaining_budget'].sum()

    def getTotalBudget(self):
        """Returns the sum of total (allocated) budgets across all subgroups."""
        return self.df['total_budget'].sum()


class StateLog(Log):
    """Tracks every layout generated while scanning one subgroup (interval between two anchors).

    Records, per generated layout, how it was derived (scan direction/order/
    value, base and increment-base layouts, predicted vs. measured
    coverage) as well as its actual page list (in the shared `pages_df`
    side-table, ``layout_pages.log``, common to all subgroups). This data
    powers the gap-finding and prediction heuristics in `layout_generator.py`
    (e.g. `getMaxGap`, `getNextIncrementBase`, `getNextExpectedRealCoverage`).

    Attributes:
        right_layout: Name of this subgroup's lower-real-coverage bounding
            anchor.
        left_layout: Name of this subgroup's higher-real-coverage bounding
            anchor.
        pages_log_name: Path to the shared ``layout_pages.log`` CSV.
        pages_df: DataFrame of ``layout``, ``base_layout``, ``added_pages``,
            ``pages`` for every layout ever created (shared across all
            `StateLog` instances via the file, though not itself a
            `Singleton`).
    """
    def __init__(self, exp_dir, results_df, right_layout, left_layout, max_gap, max_budget, dry_run):
        default_columns = [
            'layout', 'scan_base', 'increment_base',
            'scan_direction', 'scan_order', 'scan_value',
            'pebs_coverage', 'increment_real_coverage',
            'expected_real_coverage', 'real_coverage',
            'walk_cycles']
        self.right_layout = right_layout
        self.left_layout = left_layout
        state_name = right_layout + '_' + left_layout
        super().__init__(exp_dir, results_df,
                         state_name + '_state.log',
                         max_gap, max_budget, dry_run,
                         default_columns)
        super().writeRealCoverage()
        self.pages_log_name = self.exp_dir + '/layout_pages.log'
        if not os.path.isfile(self.pages_log_name):
            self.pages_df = pd.DataFrame(columns=[
                'layout', 'base_layout',
                'added_pages', 'pages'])
        else:
            self.pages_df = pd.read_csv(self.pages_log_name, converters={
                "pages": literal_eval, "added_pages": literal_eval})

    def addRecord(self,
                  layout,
                  scan_direction,
                  scan_order,
                  scan_value, scan_base,
                  pebs_coverage, expected_real_coverage, increment_base,
                  pages,
                  writeLog=True):
        """Records a newly generated layout: its scan metadata (this log) and its page list (`pages_df`).

        Args:
            layout: Name of the new layout.
            scan_direction: ``'add'``, ``'remove'``, or ``'auto'``.
            scan_order: ``'tail'``, ``'head'``, or ``'blind'``.
            scan_value: The scan's target PEBS coverage (or mixing factor,
                for blind scans).
            scan_base: Name of the base layout this one was derived from
                (``'none'``/``'other'`` for anchors).
            pebs_coverage: The new layout's PEBS-predicted coverage.
            expected_real_coverage: The real coverage this layout was
                predicted to achieve.
            increment_base: Name of the increment-base (target) layout.
            pages: The new layout's full page list.
            writeLog: If True, immediately persists this log (not
                `pages_df`, which is always written when a new layout is
                added).
        """
        base_pages = []
        if scan_base != 'none' and scan_base != 'other':
            base_pages = self.getLayoutPages(scan_base)
        added_pages = list(set(pages) - set(base_pages))
        added_pages.sort()
        new_row = pd.Series({
            'layout': layout,
            'scan_direction': scan_direction,
            'scan_order': scan_order,
            'scan_value': scan_value,
            'scan_base': scan_base,
            'pebs_coverage': pebs_coverage,
            'expected_real_coverage': expected_real_coverage,
            'increment_base': increment_base,
            'increment_real_coverage': self.getRealCoverage(increment_base),
            'real_coverage': -1,
            'walk_cycles': -1
            })
        self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        if writeLog:
            self.writeLog()
        if layout not in self.pages_df['layout']:
            new_row = pd.Series({
                'layout': layout,
                'base_layout': scan_base,
                'added_pages': added_pages,
                'pages': pages
                })
            self.pages_df = pd.concat([self.pages_df, new_row.to_frame().T], ignore_index=True)
            if not self.dry_run:
                self.pages_df.to_csv(self.pages_log_name, index=False)

    def getLayoutPages(self, layout):
        """Returns the full page list previously recorded for `layout` in `pages_df`."""
        pages = self.pages_df.loc[self.pages_df['layout'] == layout, 'pages'].iloc[0]
        return pages

    def getLayoutAddedPages(self, layout):
        """Returns the pages that were newly added to `layout` relative to its base layout."""
        return self.getField(layout, 'added_pages')

    def hasOnlyBaseLayouts(self):
        """Returns True if no layout has been generated yet for this subgroup (only anchors exist)."""
        df = self.df.query(f'scan_base != "none" and scan_base != "other"')
        return len(df) == 0

    def hasOnlyOneNewLayout(self):
        """Returns True if exactly one layout has been generated so far for this subgroup."""
        df = self.df.query(f'scan_base != "none" and scan_base != "other"')
        return len(df) == 1

    def getRightLayoutName(self):
        """Returns the name of this subgroup's lower-real-coverage bounding anchor."""
        return self.right_layout

    def getLeftLayoutName(self):
        """Returns the name of this subgroup's higher-real-coverage bounding anchor."""
        return self.left_layout

    def getRigthRecord(self):
        """Returns the log row for the right (lower-coverage) bounding anchor."""
        assert(not self.empty())
        return self.getRecord('layout', self.getRightLayoutName())

    def getLeftRecord(self):
        """Returns the log row for the left (higher-coverage) bounding anchor."""
        assert(not self.empty())
        return self.getRecord('layout', self.getLeftLayoutName())

    def getPebsCoverageDeltaBetweenLayoutAndItsBase(self, layout):
        """Returns `layout`'s PEBS coverage minus its base layout's PEBS coverage, or None if it has no base."""
        base_layout = self.getBaseLayout(layout)
        if base_layout is None or base_layout == 'none':
            return None

        layout_coverage = self.getPebsCoverage(layout)
        assert(layout_coverage is not None)
        base_coverage = self.getPebsCoverage(base_layout)
        assert(base_coverage is not None)

        delta = layout_coverage - base_coverage
        return delta

    def getGapBetweenLayoutAndItsBase(self, layout):
        """Returns the real-coverage gap between `layout` and its base layout, or None if it has no base."""
        base_layout = self.getBaseLayout(layout)
        if base_layout is None or base_layout == 'none':
            return None
        return self.getGapFromBase(layout, base_layout)

    def getGapFromBase(self, layout, base_layout):
        """Returns `layout`'s real coverage minus `base_layout`'s real coverage.

        Args:
            layout: The derived layout.
            base_layout: The layout it was derived from.

        Returns:
            float: The signed real-coverage difference.
        """
        layout_coverage = self.getRealCoverage(layout)
        assert(layout_coverage is not None)
        base_coverage = self.getRealCoverage(base_layout)
        assert(base_coverage is not None)

        gap = layout_coverage - base_coverage
        print(f'{layout} real-coverage: {layout_coverage} , {base_layout} real-coverage: {base_coverage} ==> gap: {gap}')
        return gap

    def getGapBetweenLastRecordAndIncrementBase(self):
        """Returns the real-coverage gap between the last generated layout and its increment base."""
        #self.writeRealCoverage()
        last_layout = self.getLastRecord()
        base_layout = last_layout['increment_base']
        return self.getGapFromBase(last_layout['layout'], base_layout)

    def getBaseLayout(self, layout_name):
        """Returns the name of the layout `layout_name` was derived (scanned) from."""
        return self.getField(layout_name, 'scan_base')

    def getIncBaseLayout(self, layout_name):
        """Returns the name of `layout_name`'s increment-base (target) layout."""
        return self.getField(layout_name, 'increment_base')

    def getLayoutScanOrder(self, layout_name):
        """Returns the scan order (``'tail'``/``'head'``/``'blind'``) used to create `layout_name`."""
        return self.getField(layout_name, 'scan_order')

    def getLayoutScanDirection(self, layout_name):
        """Returns the scan direction (``'add'``/``'remove'``/``'auto'``) used to create `layout_name`."""
        return self.getField(layout_name, 'scan_direction')

    def getLayoutScanValue(self, layout_name):
        """Returns the scan value (target PEBS coverage or mixing factor) used to create `layout_name`."""
        return self.getField(layout_name, 'scan_value')

    def getNextBaseLayout(self, scan_direction, scan_order):
        """Finds the best base layout to scan from next, given a scan direction/order.

        Restricts to layouts within the range bounded by the rightmost
        layout and the current increment base, whose `scan_base` is
        ``'none'`` (an anchor) or whose direction/order matches the
        requested one, and returns the one with the highest real coverage.

        Args:
            scan_direction: ``'add'`` or ``'remove'``.
            scan_order: ``'tail'`` or ``'head'``.

        Returns:
            str | None: The chosen base layout's name, or ``None`` if no
                increment base is available yet.
        """
        start_layout = self.getRightLayoutName()
        start_layout_coverage = self.getRealCoverage(start_layout)
        max_coverage = self.getRealCoverage(self.getLeftLayoutName())
        increment_base = self.getNextIncrementBase()
        if increment_base is None:
            return None
        increment_layout_coverage = self.getRealCoverage(increment_base)

        df = self.df.query(f'real_coverage >= {start_layout_coverage} and real_coverage <= {increment_layout_coverage}')
        df = df.query(f'scan_base == "none" or (scan_direction == "{scan_direction}" and scan_order == "{scan_order}")')
        df = df.sort_values('real_coverage', ascending=True)
        assert len(df) > 0
        return df.iloc[-1]['layout']

    def getNextIncrementBase(self):
        """Walks up from the rightmost layout, chaining together layouts within `max_gap` of each other, to find the current increment-base target.

        Starting at the rightmost (lowest real-coverage) layout, repeatedly
        advances to the next-higher-coverage layout as long as doing so
        stays within `max_gap` of the current position, stopping either
        when no such layout exists or when the leftmost layout's coverage
        has been reached (in which case there is no more gap to close, so
        ``None`` is returned).

        Returns:
            str | None: The name of the furthest layout reachable within
            chained `max_gap` steps, or ``None`` if the whole range is
            already closed.
        """
        start_layout = self.getRightLayoutName()
        start_layout_coverage = self.getRealCoverage(start_layout)
        max_coverage = self.getRealCoverage(self.getLeftLayoutName())
        df = self.df.query(f'real_coverage >= {start_layout_coverage}')
        df = df.sort_values('real_coverage', ascending=True)
        current_coverage = start_layout_coverage
        current_layout = start_layout
        for index, row in df.iterrows():
            if row['real_coverage'] <= (current_coverage + self.max_gap):
                current_layout = row['layout']
                current_coverage = row['real_coverage']
                if current_coverage >= max_coverage:
                    return None
            else:
                break
        return current_layout

    def getNextExpectedRealCoverage(self):
        """Predicts the real-coverage value the next generated layout should target.

        Aims for roughly ``7/8`` of `max_gap` past the current increment
        base (leaving margin below `max_gap` for measurement noise); if
        another already-measured layout exists not too far beyond that
        point, targets the midpoint between them instead.

        Returns:
            float: The target real-coverage percentage for the next layout.
        """
        expected_increment = (7.0/8.0) * self.max_gap

        next_increment = self.getNextIncrementBase()
        inc_real_coverage = self.getRealCoverage(next_increment)
        max_expected_real = inc_real_coverage + 2 * expected_increment

        df = self.df.query(f'{inc_real_coverage} < real_coverage <= {max_expected_real}')

        if len(df) == 0:
            return inc_real_coverage + expected_increment

        df = df.sort_values('real_coverage')
        upper_real_coverage = df.iloc[0]['real_coverage']
        avg_real_coverage = (inc_real_coverage + upper_real_coverage) / 2
        return avg_real_coverage

    def getMaxGapLayouts(self, include_other_layouts=True):
        """Finds the two adjacent (by real coverage) layouts within this subgroup's range with the largest gap.

        Args:
            include_other_layouts: If True, considers every layout within
                the right/left bounds; if False, excludes layouts whose
                `scan_base` is ``'other'`` (e.g. anchors imported from a
                different subgroup boundary via `updateStateLog`).

        Returns:
            tuple[str, str]: ``(right, left)`` names of the two layouts
            bounding the largest gap, with `right` having lower real
            coverage than `left`.
        """
        left_coverage = self.getRealCoverage(self.getLeftLayoutName())
        right_coverage = self.getRealCoverage(self.getRightLayoutName())
        if include_other_layouts:
            query = self.df.query(f'{right_coverage} <= real_coverage <= {left_coverage}')
        else:
            query = self.df.query(f'{right_coverage} <= real_coverage <= {left_coverage} and scan_base != "other"')
        diffs = query.sort_values('real_coverage', ascending=True)
        diffs['diff'] = diffs['real_coverage'].diff().abs()

        idx_label = diffs['diff'].idxmax()
        idx = diffs.index.get_loc(idx_label)
        right = diffs.iloc[idx-1]
        left = diffs.iloc[idx]
        return right['layout'], left['layout']

    def getMaxGap(self):
        """Returns the size (in real-coverage percentage points) of the largest gap in this subgroup."""
        right, left = self.getMaxGapLayouts()
        max_gap = abs(self.getRealCoverage(left) - self.getRealCoverage(right))
        print(f'=========> the maximal gap was found between: {right} - {left}, which is: {max_gap:.2f} <=========')
        return max_gap

    def getAllLayouts(self):
        """Returns the names of every layout recorded in this subgroup's log."""
        return self.df['layout'].to_list()
