#!/usr/bin/env python3
"""MosSelect: an adaptive, PEBS-guided hugepage layout selector.

This module implements the ``LayoutGenerator`` used by the ``moselect``
experiment (see `experiments/moselect/module.mk` and `createLayouts.py`) to
choose which Mosalloc hugepage layout to measure next, aiming to produce a
set of layouts whose runtimes are evenly spread out along the full
performance range (from the no-hugepages layout to the all-hugepages
layout), within a fixed measurement budget (`--max_budget`, typically 50
layouts). Evenly-spread samples make the resulting scatter plot / Mosmodel
regression far more informative than an arbitrary or purely random sample of
layouts would be.

Core terminology
-----------------
- **PEBS coverage** (``pebs_coverage``): the *statically predicted* fraction
  (0-100) of sampled TLB accesses that a layout's hugepage set is expected
  to serve, computed directly from a one-time Intel PEBS sampling pass (see
  `LayoutGeneratorUtils.normalizePebsAccesses`). This is a proxy that is
  cheap to compute for a candidate hugepage set, without running the
  benchmark.
- **Real coverage** (``real_coverage``): the *actually measured* runtime
  position of a layout, expressed as a percentage between the slowest
  measured layout (0%, typically the no-hugepages layout, highest
  ``walk_cycles``) and the fastest (100%, typically the all-hugepages
  layout, lowest ``walk_cycles``). This is only known after a layout has
  been executed and its `walk_cycles` measured.
- **Gap**: the difference in `real_coverage` between two layouts that are
  adjacent along the runtime axis. The algorithm's goal is to keep every
  such gap below `max_gap` (`--max_gap`), i.e., no two *consecutive*
  measured layouts should be more than `max_gap` percentage points apart in
  real coverage.
- **Subgroup**: one of the intervals between two adjacent "anchor" layouts
  produced by the initial static phase (see below); each subgroup is
  allocated a **budget** of additional layout evaluations, proportional to
  the size of its initial real-coverage gap (see
  `SubgroupsLog.calculateBudget` in `logs.py`).
- **base layout** / **increment base**: when generating a new layout inside
  a subgroup, the algorithm mixes/derives pages from a *base* (starting)
  layout and moves towards an *increment base* (target) layout, either by
  adding pages (to increase real coverage, moving towards higher hugepage
  usage) or removing pages (to decrease it).

Algorithm phases
----------------
1. **Static bootstrap** (`createInitialLayoutsStatically` /
   `createSubgroups`, used for ``layout1``-``layout9``): PEBS pages are
   greedily partitioned into three coverage-weighted buckets
   (~56%/28%/14%, see `fillBuckets`), and all ``2**3 = 8`` subset
   combinations of the three buckets are evaluated as layouts, plus one
   additional "all pages backed by 2MB hugepages" layout -- 9 anchor
   layouts total. These anchors are expected to roughly span the full
   runtime range and define the initial subgroups.
2. **Adaptive gap closing** (`createNextLayoutDynamically` and friends, used
   from ``layout10`` onward): once the anchors' real measurements are
   available, each subgroup receives a budget (`SubgroupsLog.calculateBudget`)
   and, while that budget lasts, the generator repeatedly finds the largest
   remaining real-coverage gap inside the subgroup and proposes a new layout
   meant to bisect it -- by adding pages (`createLayoutUsingScanMethod('add')`),
   removing pages (`'remove'`), or, if those data-driven strategies fail to
   produce a not-yet-tried layout, falling back to blind/heuristic page
   selection (`autoReduceMaximalGap`).
3. **Budget redistribution / final cleanup** (`findSubgroupsToRedistribute`,
   `updateLogs`, `autoReduceMaximalGap`): if a subgroup's real gaps turn out
   far larger than PEBS predicted, its budget can be redistributed
   (`redistributeSubgroup`); once every subgroup's budget is exhausted, any
   leftover evaluations are spent closing the single largest remaining gap
   across all subgroups.

Persistent state (the `subgroups.log`, `<right>_<left>_state.log`, and
`layout_pages.log` CSV files under the experiment directory) is managed by
`logs.py`'s `SubgroupsLog`/`StateLog` classes, which also makes the whole
process resumable across independent `make`-triggered invocations (this
script is invoked once per layout to generate; see `module.mk`).
"""
from struct import calcsize
import sys
import os
import collections
import pandas as pd
import itertools
import os.path
from logs import *

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from Utils.utils import Utils
from Utils.ConfigurationFile import Configuration

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../analysis')
from performance_statistics import PerformanceStatistics

# Pages whose individual PEBS TLB-coverage is at least this percentage are
# treated as "heavy" (head) pages, and are considered separately/combinatorially
# by the addMinimalHeadPages* family below, since a single such page can shift
# the total coverage by a large, hard-to-fine-tune amount.
HEAD_PAGES_WEIGHT_THRESHOLD = 5.0

class LayoutGenerator():
    """Generates the next Mosalloc hugepage layout to evaluate for one benchmark.

    A fresh `LayoutGenerator` is constructed once per `make`-triggered
    invocation of `createLayouts.py` (i.e., once per layout to be produced);
    all cross-invocation state is persisted to disk via `SubgroupsLog` and
    `StateLog` (see `logs.py`) and reloaded here through
    `getAllLayoutsFromStateLogs`.

    Attributes:
        pebs_df: DataFrame of per-page PEBS TLB-coverage
            (``PAGE_NUMBER``/``TLB_COVERAGE``), from
            `LayoutGeneratorUtils.normalizePebsAccesses`.
        results_df: DataFrame of all layouts measured so far (``layout``,
            ``walk_cycles``, ...), or ``None`` if no results exist yet.
        layout: Name of the layout this invocation is responsible for
            producing (e.g. ``'layout12'``).
        exp_dir: Root directory of this experiment (layouts, logs, and
            results live under here).
        max_gap: Maximum acceptable real-coverage gap (percentage points)
            between two adjacent measured layouts.
        default_increment: Default PEBS-coverage step used as a fallback
            when scaling/predicting the next coverage target (``2 * max_gap``).
        max_budget: Total number of layouts allowed for this experiment.
        debug: If True, layouts are printed but not actually written to disk
            (dry run).
        subgroups_log: The `SubgroupsLog` tracking each anchor subgroup's
            budget and coverage.
        state_log: The `StateLog` for the subgroup currently being processed
            (set lazily; `None` until the first subgroup is selected).
    """
    def __init__(self, pebs_df, results_df, layout, exp_dir, max_gap, max_budget, debug):
        self.pebs_df = pebs_df
        self.results_df = results_df
        self.layout = layout
        self.exp_dir = exp_dir
        self.max_gap = max_gap
        self.default_increment = 2 * max_gap
        self.max_budget = max_budget
        self.debug = debug
        self.subgroups_log = SubgroupsLog(exp_dir, results_df, max_gap, max_budget, debug)
        self.all_layouts = self.getAllLayoutsFromStateLogs()
        self.state_log = None
    def generateLayout(self):
        """Produces `self.layout`, dispatching to the static or adaptive phase.

        - ``layout1``: runs the one-time static bootstrap (9 anchor layouts).
        - ``layout10``: first checks whether any subgroup's *real* gap turned
          out much larger than its PEBS-predicted gap and, if so,
          redistributes that subgroup's budget into extra sub-anchor layouts
          (`findSubgroupsToRedistribute`) before falling through to the
          adaptive phase.
        - Any other layout name: proceeds directly to the adaptive
          gap-closing phase (`createNextLayoutDynamically`).
        """
        if self.layout == 'layout1':
            # 1.1. create nine layouts statically (using PEBS output):
            self.createInitialLayoutsStatically()
            return
        if self.layout == 'layout10':
            if self.findSubgroupsToRedistribute():
                return
        # 1.2. create other layouts dynamically
        self.createNextLayoutDynamically()

    def createInitialLayoutsStatically(self):
        """Runs the static bootstrap phase: produces the 9 anchor layouts.

        Partitions PEBS pages into three coverage-weighted buckets
        (~56%/28%/14%) via `fillBuckets`, then materializes all subset
        combinations of the buckets as layouts (see `createSubgroups`).
        """
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fillBuckets(self.pebs_df, buckets_weights)
        self.createSubgroups(group)

    def fillBuckets(self, df, buckets_weights, start_from_tail=False, fill_min_buckets_first=True):
        """Greedily partitions PEBS pages into buckets matching target coverage weights.

        Iterates over pages sorted by `TLB_COVERAGE` (descending by default,
        or ascending if `start_from_tail`) and assigns each page to the
        bucket whose remaining capacity best fits the page's weight (or
        worst-fits, if `fill_min_buckets_first` is False), so that each
        bucket's assigned pages approach its target weight in
        `buckets_weights`.

        Args:
            df: PEBS DataFrame with ``PAGE_NUMBER``/``TLB_COVERAGE`` columns
                to select pages from.
            buckets_weights: List of target TLB-coverage percentages, one
                per bucket; consumed in place (decremented as pages are
                assigned).
            start_from_tail: If True, considers pages from lowest to highest
                TLB coverage instead of the default highest-to-lowest.
            fill_min_buckets_first: If True, prefers the bucket with the
                least remaining capacity that can still fit the page
                (best-fit); if False, prefers the bucket with the most
                remaining capacity (worst-fit).

        Returns:
            list[list[int]]: One list of PEBS page numbers per bucket, in
            the same order as `buckets_weights`.
        """
        group_size = len(buckets_weights)
        group = [ [] for _ in range(group_size) ]
        df = df.sort_values('TLB_COVERAGE', ascending=start_from_tail)

        threshold = 2
        i = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            selected_weight = None
            selected_index = None
            completed_buckets = 0
            # count completed buckets and find bucket with minimal remaining
            # space to fill, i.e., we prefer to place current page in the
            # bicket that has the lowest remaining weight/space
            for i in range(group_size):
                if buckets_weights[i] <= 0:
                    completed_buckets += 1
                elif buckets_weights[i] >= weight - threshold:
                    if selected_index is None:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif fill_min_buckets_first and buckets_weights[i] < selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif not fill_min_buckets_first and buckets_weights[i] > selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
            if completed_buckets == group_size:
                break
            # if there is a bucket that has a capacity of current page, add it
            if selected_index is not None:
                group[selected_index].append(page)
                buckets_weights[selected_index] -= weight
        return group

    def writeLayoutAll2mb(self, layout_name, output):
        """Writes the trivial "all pages backed by 2MB hugepages" layout (100% coverage)."""
        if not self.debug:
            print(layout_name)
            print('weight: 100%')
            print('hugepages: all pages')
            LayoutGeneratorUtils.writeLayoutAll2mb(layout_name, output)

    def writeLayout(self, layout_name, pages):
        """Writes a Mosalloc layout configuration for the given hugepage set.

        Args:
            layout_name: Name of the layout (e.g. ``'layout7'``), used for
                both the printed summary and the output CSV file name.
            pages: List of PEBS/base-hugepage page numbers to back with
                hugepages in this layout.

        Returns:
            float: The PEBS-predicted TLB coverage of `pages`.
        """
        total_pages = len(self.pebs_df)
        pebs_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, pages)
        print(layout_name)
        pages_ratio=round(len(pages)/total_pages * 100)
        print(f'#hugepages: {len(pages)} (~{pages_ratio}%) out of {total_pages} pages (reported by PEBS)')
        print(f'weight: {pebs_coverage}')
        print(f'hugepages: {pages}')
        print('---------------------------------------------')
        if not self.debug:
            LayoutGeneratorUtils.writeLayout(layout_name, pages, self.exp_dir)
        return pebs_coverage

    def createSubgroups(self, group):
        """Materializes every subset of the three coverage buckets as a layout.

        Given the three page buckets produced by `fillBuckets`, evaluates all
        ``2**3 = 8`` subset combinations (including the empty and full
        subsets) as ``layout1``..``layout8``, then adds a 9th layout
        (``layout9``) with every page backed by a 2MB hugepage. Each
        generated layout is recorded in `subgroups_log` as an anchor for the
        adaptive phase.

        Args:
            group: The three page buckets returned by `fillBuckets`.
        """
        i = 1
        # 1.1.2. create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                subset_pages = list(itertools.chain(*subset))
                layout_name = f'layout{i}'
                pebs_coverage = self.writeLayout(layout_name, subset_pages)
                i += 1
                self.subgroups_log.addRecord(layout_name, pebs_coverage)
        # 1.1.3. create additional layout in which all pages are backed with 2MB
        layout_name = f'layout{i}'
        self.writeLayoutAll2mb(layout_name, self.exp_dir)
        self.subgroups_log.addRecord(layout_name, 100)
        self.subgroups_log.writeLog()

    def findSubgroupsToRedistribute(self):
        """Redistributes budget for anchor subgroups whose real gap far exceeds `max_gap`.

        Called once (for ``layout10``) after all 9 static anchors have been
        measured. For every pair of adjacent anchors whose *real*-coverage
        gap exceeds `real_coverage_threshold` (20 percentage points), creates
        additional "sub-anchor" layouts inside that pair via
        `redistributeSubgroup`, giving the adaptive phase a better-populated
        starting point for that subgroup.

        Returns:
            bool: True if any new layouts were created (so the caller should
            stop and let `make` measure them before proceeding further).
        """
        real_coverage_threshold = 20
        self.updateSubgroupsLog(False)
        next_layout_num = 10
        created_new_layouts = False
        for i in range(len(self.subgroups_log.df)-1):
            right, left = self.subgroups_log.getSubgroup(i)
            right_layout = right['layout']
            left_layout = left['layout']
            real_coverage_delta = left['real_coverage'] - right['real_coverage']
            if real_coverage_delta > real_coverage_threshold:
                next_layout_num = self.redistributeSubgroup(right_layout, left_layout, next_layout_num)
                created_new_layouts = True
        return created_new_layouts

    def redistributeSubgroup(self, right, left, start_layout_number):
        """Creates sub-anchor layouts between two anchors with an oversized real gap.

        Splits the PEBS-coverage range spanned by `right`/`left` into three
        new coverage-weighted buckets (mid-point-based) and evaluates all
        pairwise subset combinations as new layouts, analogous to
        `createSubgroups` but scoped to this one interval.

        Args:
            right: Name of the lower-real-coverage anchor layout.
            left: Name of the higher-real-coverage anchor layout.
            start_layout_number: First layout number to use for the newly
                created layouts (e.g. ``10``).

        Returns:
            int: The next unused layout number after this call.
        """
        right_pages = LayoutGeneratorUtils.getLayoutHugepages(right, self.exp_dir)
        left_pages = LayoutGeneratorUtils.getLayoutHugepages(left, self.exp_dir)
        right_pebs = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, right_pages)
        left_pebs = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, left_pages)

        # calculate the desired weights to distribute the new layout according to
        pebs_min = min(right_pebs, left_pebs)
        pebs_max = max(right_pebs, left_pebs)
        pebs_avg = (pebs_min + pebs_max) / 2
        weights = [pebs_min/2, pebs_avg/2, pebs_max/2]

        pages_group = self.fillBuckets(self.pebs_df, weights)

        layout_number = start_layout_number
        for subset in itertools.combinations(pages_group, 2):
            subset_pages = list(itertools.chain(*subset))
            layout_name = f'layout{layout_number}'
            pebs_coverage = self.writeLayout(layout_name, subset_pages)
            layout_number += 1
            self.subgroups_log.addRecord(layout_name, pebs_coverage)
        self.subgroups_log.writeLog()

        return layout_number

    def updateSubgroupsLog(self, calculateBudget=True):
        """Refreshes `subgroups_log` with real-coverage measurements and (optionally) budgets.

        On the first call (log still empty), seeds `subgroups_log` from
        `results_df` (assumed to hold the 9 static anchors' measurements at
        this point). On every call, recomputes `real_coverage` for any
        not-yet-measured rows (`Log.writeRealCoverage`).

        Args:
            calculateBudget: If True, (re)computes each subgroup's
                measurement budget (`SubgroupsLog.calculateBudget`); if
                False, only re-sorts the log by real coverage.
        """
        # calculate the real-coverage for each group and update the log
        # if the subgroups-log was not created yet then create it based on the
        # current results
        #subgroups_layouts = ['layout1', 'layout2','layout3', 'layout4','layout5','layout6','layout7','layout8','layout9']
        if self.subgroups_log.empty():
            #results_df_sorted = self.results_df.query(
            #        f'layout in {subgroups_layouts}').sort_values(
            #                'walk_cycles', ascending=False)
            results_df_sorted = self.results_df.sort_values('walk_cycles', ascending=False)
            for index, row in results_df_sorted.iterrows():
                layout = row['layout']
                layout_pages = LayoutGeneratorUtils.getLayoutHugepages(layout, self.exp_dir)
                layout_pebs = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, layout_pages)
                self.subgroups_log.addRecord(layout, layout_pebs)
            self.subgroups_log.writeRealCoverage()
            self.subgroups_log.df = self.subgroups_log.df.sort_values('real_coverage')
            self.subgroups_log.writeLog()
        else:
            self.subgroups_log.writeRealCoverage()
        if calculateBudget:
            # calculate the budget that will be given for each group
            self.subgroups_log.calculateBudget()
        else:
            self.subgroups_log.sortByRealCoverage()

    def getSubgroupWithMaximalGap(self):
        """Finds the subgroup with the single largest current real-coverage gap.

        Iterates over all adjacent anchor pairs, builds/updates each pair's
        `StateLog`, and returns the one whose largest internal gap
        (`StateLog.getMaxGap`) is the biggest -- used as a fallback target
        once every subgroup's own budget has been exhausted.

        Returns:
            tuple[float, StateLog]: The maximal gap value and its
            corresponding `StateLog`.
        """
        max_gap = 0
        state_log = None
        # find the first group that still has a remaining budget
        for i in range(len(self.subgroups_log.df)-1):
            right, left = self.subgroups_log.getSubgroup(i)
            right_layout = right['layout']
            left_layout = left['layout']
            # initialize the state-log for the current group
            self.state_log = StateLog(self.exp_dir,
                                      self.results_df,
                                      right_layout,
                                      left_layout,
                                      self.max_gap,
                                      self.max_budget,
                                      self.debug)
            # if the state log is empty then it seems just now we are
            # about to start scanning this group
            self.updateStateLog(right, left)
            curr_max_gap = self.state_log.getMaxGap()
            if curr_max_gap > max_gap:
                max_gap = curr_max_gap
                state_log = self.state_log
        return max_gap, state_log

    def getFirstSubgroupToProcess(self):
        """Finds the first subgroup (left to right) that still has budget and open gaps.

        Iterates over adjacent anchor pairs in order and returns the first
        one that both (a) still has a gap to close
        (`StateLog.getNextIncrementBase` is not None) and (b) has remaining
        budget. Subgroups that already closed all their gaps are skipped;
        subgroups that ran out of budget without closing all gaps are
        counted in `unclosed_subgroups` (used by
        `initStateLogForNextSubgroupToProcess` to decide whether to borrow
        budget from elsewhere).

        Returns:
            tuple[StateLog | None, int]: The `StateLog` of the first
            processable subgroup (or ``None`` if none qualify), and the
            count of subgroups that exhausted their budget with gaps still
            open.
        """
        unclosed_subgroups = 0
        # find the first group that still has a remaining budget
        for i in range(len(self.subgroups_log.df)-1):
            right, left = self.subgroups_log.getSubgroup(i)
            right_layout = right['layout']
            left_layout = left['layout']
            # initialize the state-log for the current group
            self.state_log = StateLog(self.exp_dir,
                                      self.results_df,
                                      right_layout,
                                      left_layout,
                                      self.max_gap,
                                      self.max_budget,
                                      self.debug)
            # if the state log is empty then it seems just now we are
            # about to start scanning this group
            self.updateStateLog(right, left)
            # if we already closed all gaps in this group then move the
            # left budget to the next group
            next_layout = self.state_log.getNextIncrementBase()
            remaining_budget = self.subgroups_log.getRemainingBudget(left_layout)
            if next_layout is None:
                print('===========================================================')
                print(f'[DEBUG] closed all gaps for subgroup: {right_layout} - {left_layout}')
                print('===========================================================')
                continue
            elif remaining_budget <= 0:
                assert remaining_budget == 0
                print('===========================================================')
                print(f'[DEBUG] consumed all budget but did not close all gaps for subgroup: {right_layout} - {left_layout}')
                print('===========================================================')
                unclosed_subgroups += 1
                continue
            else:
                assert remaining_budget > 0
                return self.state_log, unclosed_subgroups
        return None, unclosed_subgroups

    def initStateLogForNextSubgroupToProcess(self):
        """Selects which subgroup `self.state_log` should point to next.

        Prefers the first subgroup that still has its own remaining budget
        (`getFirstSubgroupToProcess`). If none remain but some subgroups
        still have open gaps (ran out of budget before closing them),
        reassigns *all* remaining budget across subgroups
        (`SubgroupsLog.zeroAllBudgets`) to the subgroup with the largest
        outstanding gap (`getSubgroupWithMaximalGap`), guaranteeing forward
        progress is always possible as long as any budget remains.

        Returns:
            bool: True if a subgroup with remaining budget was found and
            `self.state_log` now points to it; False if every subgroup's
            gaps are already within `max_gap`.
        """
        state_log, unclosed_subgroups = self.getFirstSubgroupToProcess()
        if state_log is None:
            if unclosed_subgroups == 0:
                print('===========================================================')
                print(f'[DEBUG] ++++ closed all gaps for all subgroups ++++')
                print('===========================================================')
                return False
            max_gap, state_log = self.getSubgroupWithMaximalGap()
            self.state_log = state_log
            print('===========================================================')
            print(f'[DEBUG] start closing gaps for subgroup: {state_log.getRightLayoutName()} - {state_log.getLeftLayoutName()}, which has the maximal gap: {max_gap}')
            print('===========================================================')
            remaining_budget = self.subgroups_log.zeroAllBudgets()
            if remaining_budget == 0:
                print('===========================================================')
                print(f'[WARNING] Consumed the total allocated budget but got a request to create a new layout!')
                print('===========================================================')
                remaining_budget = 1
            self.subgroups_log.addExtraBudget(state_log.getLeftLayoutName(), remaining_budget)
        self.state_log = state_log
        return True

    def updateLogs(self):
        """Refreshes all log state and decides whether there is still work to do.

        Refreshes `subgroups_log` (`updateSubgroupsLog`) and tries to select
        the next subgroup to process (`initStateLogForNextSubgroupToProcess`).
        If every subgroup's gaps are already closed, falls back to spending
        any leftover budget on the single largest remaining gap across *all*
        subgroups combined (`autoReduceMaximalGap`), treating the full
        left-to-right layout range as one state log.

        Returns:
            bool: True if a specific subgroup still needs a new layout (the
            caller should proceed to `createNextLayoutDynamically`'s normal
            scan methods); False if this call already produced the final
            layout via `autoReduceMaximalGap` (no further action needed).
        """
        self.updateSubgroupsLog()

        # if there is a subgroup that still has gaps to close, then process it
        if self.initStateLogForNextSubgroupToProcess():
            return True
        # otherwise (all gaps were closed), then move to minimize the maximal gap

        extra_budget = self.subgroups_log.getExtraBudget()
        print(f'finished the last group but there is still ({extra_budget}) remaining budget.')
        print('using the remaining budget to close remaining gaps in previous groups')
        right = self.subgroups_log.getRightmostLayout()
        left = self.subgroups_log.getLeftmostLayout()

        # use the extra budget one by one
        if extra_budget > 0:
            self.subgroups_log.addExtraBudget(left['layout'], 1)

        # define a new state-log that contains all layouts in all subgroups
        self.state_log = StateLog(self.exp_dir,
                                    self.results_df,
                                    right['layout'], left['layout'],
                                    self.max_gap, self.max_budget, self.debug)
        self.updateStateLog(right, left)
        self.autoReduceMaximalGap()
        return False

    def mixLayoutPagesByFactor(self, left, right, factor):
        """Blends the page sets of two layouts using a fixed sampling factor.

        Starting from the pages common to both `left` and `right`, adds
        ``1/factor`` of the pages unique to the larger ("max") layout and
        ``(factor-1)/factor`` of the pages unique to the smaller ("min")
        layout (or the reverse, if `factor` < 1), producing a blind (not
        coverage-targeted) intermediate layout. Used as a last-resort
        strategy by `autoReduceMaximalGapByFactor` when coverage-targeted
        page selection fails to produce a new, not-yet-tried layout.

        Args:
            left: Name of one of the two layouts to blend.
            right: Name of the other layout to blend.
            factor: Sampling factor (see above); may be a fraction to
                reverse which layout is favored.

        Returns:
            tuple[list[int], float]: The blended page set and its
            PEBS-predicted coverage.
        """
        reverse_order = factor < 1
        if reverse_order:
            factor = 1/factor
        factor = int(factor)

        left_pages = LayoutGeneratorUtils.getLayoutHugepages(left, self.exp_dir)
        right_pages = LayoutGeneratorUtils.getLayoutHugepages(right, self.exp_dir)
        if len(left_pages) > len(right_pages) and not reverse_order:
            max_pages = left_pages
            min_pages = right_pages
            max_layout = left
            min_layout = right
        else:
            min_pages = left_pages
            max_pages = right_pages
            min_layout = left
            max_layout = right

        common_pages = list(set(max_pages) & set(min_pages))
        only_max_pages = list(set(max_pages) - set(common_pages))
        only_min_pages = list(set(min_pages) - set(common_pages))
        # sort pages by coverage to select pages in a balanced way as much as possible
        to_be_added_from_min = self.pebs_df.query(f'PAGE_NUMBER in {only_min_pages}').sort_values('TLB_COVERAGE', ascending=False)['PAGE_NUMBER'].to_list()
        to_be_added_from_max = self.pebs_df.query(f'PAGE_NUMBER in {only_max_pages}').sort_values('TLB_COVERAGE', ascending=False)['PAGE_NUMBER'].to_list()
        # add pages that are not captured by PEBS
        to_be_added_from_min += list(set(only_min_pages) - set(to_be_added_from_min))
        to_be_added_from_max += list(set(only_max_pages) - set(to_be_added_from_max))
        # select pages by the given factor:
        # 1/factor pages from the max-layout
        added_from_max = to_be_added_from_max[::factor]
        # and (factor-1)/factor pages from the min-layout
        added_from_min = list(set(to_be_added_from_min) - set(to_be_added_from_min[::factor]))
        # drop duplicated pages and combine min and max sets
        mixed_pages = list(set(added_from_min + added_from_max + common_pages))
        mixed_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, mixed_pages)

        print(f'[DEBUG]: mixLayoutPagesByFactor - left: {left} , right: {right} , factor: {factor}')
        print(f'\t added {len(added_from_min)} pages (out of {len(min_pages)}) from {min_layout}')
        print(f'\t added {len(added_from_max)} pages (out of {len(max_pages)}) from {max_layout}')
        print(f'\t new total pages: {len(mixed_pages)}')
        print(f'\t new pebs coverage: {mixed_coverage}')

        return mixed_pages, mixed_coverage

    def moveToAnotherStateLog(self, right, left):
        """Repoints `self.state_log` at a different (right, left) layout pair.

        Used when the currently tracked pair is no longer the most useful
        one to scan (e.g., the left layout already reached ~100% PEBS
        coverage without reaching ~100% real coverage, so a wider range
        including the true all-hugepages layout is needed instead).

        Args:
            right: The new lower-coverage bound layout record (a `Series`
                with at least a ``'layout'`` field).
            left: The new upper-coverage bound layout record.
        """
        right_layout = right['layout']
        left_layout = left['layout']
        print(f'[DEBUG]: Moving to use a new state log: {left_layout} - {right_layout}')
        # define a new state-log that contains all layouts in all subgroups
        self.state_log = StateLog(self.exp_dir,
                                    self.results_df,
                                    right_layout, left_layout,
                                    self.max_gap, self.max_budget, self.debug)
        self.updateStateLog(right, left)


    def autoReduceMaximalGap(self):
        """Last-resort strategy: blindly bisects the current largest real-coverage gap.

        Tries, in order, three increasingly desperate strategies to produce a
        new, not-yet-tried layout that lies between the two layouts with the
        largest gap: mixing by a fixed factor
        (`autoReduceMaximalGapByFactor`), mixing to target the midpoint PEBS
        coverage (`autoReduceMaximalGapByCoverage`), and finally recursively
        removing pages (`removePagesRecursively`). Whichever succeeds first
        is written out as `self.layout`.

        Returns:
            bool: Always True (a layout is always eventually produced,
            barring an assertion failure if all strategies fail).
        """
        base_layout, inc_base, factor, pages, pebs_coverage = self.autoReduceMaximalGapByFactor()
        if pages is None or self.pagesSetExist(pages):
            base_layout, inc_base, factor, pages, pebs_coverage = self.autoReduceMaximalGapByCoverage()
        if pages is None or self.pagesSetExist(pages):
            base_layout, inc_base, factor, pages, pebs_coverage = self.removePagesRecursively()
        assert pages is not None
        expected_real_coverage = (self.state_log.getRealCoverage(base_layout) + self.state_log.getRealCoverage(inc_base)) / 2
        self.writeLayout(self.layout, pages)
        self.state_log.addRecord(self.layout, 'auto', 'reduce-max',
                                 factor, base_layout,
                                 pebs_coverage, expected_real_coverage,
                                 inc_base, pages)
        # decrease current group's budget by 1
        self.subgroups_log.decreaseRemainingBudget(
            self.state_log.getLeftLayoutName())
        return True


    def autoReduceMaximalGapByFactor(self):
        """Attempts to bisect the current largest gap via `mixLayoutPagesByFactor`.

        Picks the mixing `factor` adaptively based on whether the previous
        attempt (if any, using the same base/increment-base pair and this
        same strategy) over- or under-shot its expected real coverage,
        doubling/halving the factor (or switching to the reversed-order
        fractional regime) accordingly to converge faster.

        Returns:
            tuple[str, str, float, list[int] | None, float]: ``(base_layout,
            inc_layout, factor, pages, pebs_coverage)`` -- `pages` is
            ``None`` if the resulting page set was empty or a duplicate of
            an existing layout.
        """
        right, left = self.state_log.getMaxGapLayouts()
        # if the left layout is with 100 pebs coverage but it's not
        # the all-2MB layout, then move to use the all-2MB layout instead
        # (which has more hugepages for sure)
        left_pebs = self.state_log.getPebsCoverage(left)
        left_real = self.state_log.getRealCoverage(left)
        if left_pebs >= 99.9 and left_real < 99.9:
            left = self.subgroups_log.getLeftmostLayout()
            right = self.state_log.getRecord('layout', right)
            self.moveToAnotherStateLog(right, left)

        print(self.state_log.df)
        right, left = self.state_log.getMaxGapLayouts()
        max_gap = abs(self.state_log.getRealCoverage(right) - self.state_log.getRealCoverage(left))
        print(f'[DEBUG]: >>>>>>>>>> current max-gap: {max_gap} by layouts: {right}-{left} <<<<<<<<<<')

        base_layout = left
        inc_layout = right
        last_layout = self.state_log.getLastLayoutName()
        last_base = self.state_log.getBaseLayout(last_layout)
        last_inc = self.state_log.getIncBaseLayout(last_layout)
        last_direction = self.state_log.getLayoutScanDirection(last_layout)
        last_order = self.state_log.getLayoutScanOrder(last_layout)
        last_factor = self.state_log.getLayoutScanValue(last_layout)
        last_real_coverage = self.state_log.getRealCoverage(last_layout)
        last_expected_real_coverage = self.state_log.getExpectedRealCoverage(last_layout)

        factor = 2
        if base_layout == last_base and inc_layout == last_inc and last_direction == 'auto' and last_order == 'reduce-max':
            if last_real_coverage < last_expected_real_coverage:
                if last_factor < 1: #use revered factor
                    factor = last_factor / 2
                else:
                    factor = last_factor * 2
            else:
                # if this was the first shot, then we need to consider
                # reversing the addition order: adding more pages from
                # the min-layout instead of the max-layout.
                # We will keep tracking this step by using factor as a fraction (=1/factor)
                if last_factor == 2: #first layout that should go in reverse order
                    factor = 0.5
                elif last_factor < 1: #reverse order for next layouts after the first one
                    factor = last_factor / 0.75
                else: # normal factor (>= 2)
                    factor = int(last_factor * 0.75)

        pages, pebs_coverage = self.mixLayoutPagesByFactor(left, right, factor)
        if pages is None or self.pagesSetExist(pages):
            pages = None
            pebs_coverage = -1
        return base_layout, inc_layout, factor, pages, pebs_coverage

    def mixLayoutPagesByCoverage(self, left, right, expected_pebs, append_pages_not_in_pebs=True):
        """Blends two layouts' pages to target a specific PEBS coverage value.

        Combines `left`'s and `right`'s pages, then greedily selects a subset
        (sorted by PEBS coverage, heaviest first) whose cumulative coverage
        is as close as possible to `expected_pebs` (within 0.5 points).
        Optionally appends any pages present in the base layouts but not
        sampled by PEBS (`append_pages_not_in_pebs`), since those contribute
        to `real_coverage` but not to the PEBS-based selection.

        Args:
            left: Name of one base layout.
            right: Name of the other base layout.
            expected_pebs: Target cumulative PEBS coverage percentage.
            append_pages_not_in_pebs: Whether to also include pages from
                `left`/`right` that PEBS did not sample.

        Returns:
            tuple[list[int] | None, float]: The blended page set (or
            ``None`` if no subset matched `expected_pebs` within tolerance)
            and its resulting PEBS coverage.
        """
        print(f'[DEBUG]: mixLayoutPagesByCoverage - left: {left} , right: {right} , expected_pebs: {expected_pebs}, add-pages-not-in-pebs: {append_pages_not_in_pebs}')
        left_pages = LayoutGeneratorUtils.getLayoutHugepages(left, self.exp_dir)
        right_pages = LayoutGeneratorUtils.getLayoutHugepages(right, self.exp_dir)
        combined_pages = list(set(left_pages + right_pages))

        # sort pages by coverage to select pages in a balanced way as much as possible
        sorted_pebs_df = self.pebs_df.query(f'PAGE_NUMBER in {combined_pages}').sort_values('TLB_COVERAGE', ascending=False)
        epsilon = 0.5
        total_pebs = 0
        pages = []
        for idx, row in sorted_pebs_df.iterrows():
            page_num = row['PAGE_NUMBER']
            coverage = row['TLB_COVERAGE']
            if (total_pebs + coverage) <= (expected_pebs + epsilon):
                pages.append(page_num)
                total_pebs += coverage
        if total_pebs < expected_pebs - epsilon or total_pebs > expected_pebs + epsilon:
            return None, -1

        # add pages that are not captured by PEBS
        left_pebs_pages = self.pebs_df.query(f'PAGE_NUMBER in {left_pages}')['PAGE_NUMBER'].to_list()
        right_pebs_pages = self.pebs_df.query(f'PAGE_NUMBER in {right_pages}')['PAGE_NUMBER'].to_list()
        left_pages_not_in_pebs = list(set(left_pages) - set(left_pebs_pages))
        right_pages_not_in_pebs = list(set(right_pages) - set(right_pebs_pages))
        pages_not_in_pebs = list(set(left_pages_not_in_pebs + right_pages_not_in_pebs))

        mixed_pages = pages
        if append_pages_not_in_pebs:
            mixed_pages += pages_not_in_pebs
        mixed_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, mixed_pages)

        print(f'\t added {len(pages)} pages captured by PEBS + {len(pages_not_in_pebs)} pages were not captured by PEBS')
        print(f'\t new total pages: {len(mixed_pages)}')
        print(f'\t new pebs coverage: {mixed_coverage}')

        return mixed_pages, mixed_coverage

    def autoReduceMaximalGapByCoverage(self):
        """Attempts to bisect the current largest gap by targeting its midpoint PEBS coverage.

        Similar in spirit to `autoReduceMaximalGapByFactor`, but instead of a
        fixed mixing factor, computes the expected midpoint PEBS coverage
        between the two gap-bounding layouts (adjusted based on the previous
        attempt's over-/under-shoot, if applicable) and tries a small set of
        epsilon offsets around it via `mixLayoutPagesByCoverage` until a
        novel page set is found.

        Returns:
            tuple[str, str, float, list[int] | None, float]: ``(base_layout,
            inc_layout, factor=2, pages, pebs_coverage)``.
        """
        print(self.state_log.df)
        right, left = self.state_log.getMaxGapLayouts()
        max_gap = abs(self.state_log.getRealCoverage(right) - self.state_log.getRealCoverage(left))
        print(f'[DEBUG]: >>>>>>>>>> current max-gap: {max_gap} by layouts: {right}-{left} <<<<<<<<<<')

        base_layout = left
        inc_layout = right
        left_pebs = self.state_log.getPebsCoverage(left)
        right_pebs = self.state_log.getPebsCoverage(right)
        expected_pebs = (left_pebs + right_pebs) / 2

        last_layout = self.state_log.getLastLayoutName()
        last_base = self.state_log.getBaseLayout(last_layout)
        last_inc = self.state_log.getIncBaseLayout(last_layout)
        last_direction = self.state_log.getLayoutScanDirection(last_layout)
        last_order = self.state_log.getLayoutScanOrder(last_layout)
        last_pebs_coverage = self.state_log.getPebsCoverage(last_layout)
        last_real_coverage = self.state_log.getRealCoverage(last_layout)
        last_expected_real_coverage = self.state_log.getExpectedRealCoverage(last_layout)

        if base_layout == last_base and inc_layout == last_inc and last_direction == 'auto' and last_order == 'reduce-max':
            if last_real_coverage < last_expected_real_coverage:
                expected_pebs = (last_pebs_coverage + left_pebs) / 2
            else:
                expected_pebs = (last_pebs_coverage + right_pebs) / 2

        epsilons = [0, 0.5, -0.5, 1, 1.5, 2]
        #epsilons = [0]+ list(itertools.chain(*[[i/10, -i/10] for i in range(5, 26, 5)]))
        for eps in epsilons:
            expected_pebs_with_epsilon = expected_pebs + eps
            pages, pebs_coverage = self.mixLayoutPagesByCoverage(left, right, expected_pebs_with_epsilon)
            if pages is not None and not self.pagesSetExist(pages):
                break

        if pages is None or self.pagesSetExist(pages):
            pages = None
            pebs_coverage = -1
        return base_layout, inc_layout, 2, pages, pebs_coverage

    def updateStateLog(self, right_layout, left_layout):
        """Registers, in `self.state_log`, every already-measured layout within a coverage range.

        Finds all rows of `results_df` whose `walk_cycles` fall between
        `left_layout` and `right_layout` (inclusive) and adds any that are
        not yet present in `self.state_log`, computing each one's PEBS
        coverage. This keeps `state_log` in sync with layouts that were
        measured but not necessarily created by the currently active
        strategy (e.g., anchors from a different subgroup boundary).

        Args:
            right_layout: Row (``Series``) of the lower-`walk_cycles` bound
                layout.
            left_layout: Row (``Series``) of the higher-`walk_cycles` bound
                layout.
        """
        # if the state was not created yet then create it and add all
        # layouts that in the range [left_layout - right_layout]
        state_layouts = self.results_df.query(
            'walk_cycles >= {left} and walk_cycles <= {right}'.format(
                left=left_layout['walk_cycles'],
                right=right_layout['walk_cycles']))
        state_layouts = state_layouts.sort_values('walk_cycles', ascending=False)
        #for layout_name in [right_layout['layout'], left_layout['layout']]:
        for index, row in state_layouts.iterrows():
            layout_name = row['layout']
            if self.state_log.layoutExist(layout_name):
                continue
            pages = LayoutGeneratorUtils.getLayoutHugepages(
                layout_name, self.exp_dir)
            pebs_coverage = LayoutGeneratorUtils.calculateTlbCoverage(
                self.pebs_df, pages)
            base = 'other'
            if layout_name == self.state_log.getRightLayoutName() or layout_name == self.state_log.getLeftLayoutName():
                base = 'none'
            self.state_log.addRecord(layout_name,
                                     'none', 'none', -1, base,
                                     pebs_coverage, -1, 'none',
                                     pages)
        self.state_log.writeLog()
        self.state_log.writeRealCoverage()

    def getWorkingSetPages(self):
        """Partitions all known pages relative to the current subgroup's two bounding layouts.

        Given the current state log's right (lower-coverage) and left
        (higher-coverage) bounding layouts, computes:
          - pages only in the right layout (candidates to *remove* when
            moving from right towards left),
          - pages only in the left layout (candidates to *add* when moving
            from right towards left),
          - the complement of the union of both layouts' pages (out of
            either layout; candidates for blind exploration), and
          - the full set of all known pages (union of both layouts and all
            PEBS-sampled pages).

        Returns:
            tuple[list[int], list[int], list[int], list[int]]: ``(alpha,
            beta, gamma, U)`` as defined in `createLayoutUsingScanMethod`'s
            docstring: ``alpha`` = right-only, ``beta`` = left-only,
            ``gamma`` = outside the union, ``U`` = all pages.
        """
        right_layout = self.state_log.getRigthRecord()['layout']
        left_layout = self.state_log.getLeftRecord()['layout']

        right = LayoutGeneratorUtils.getLayoutHugepages(right_layout, self.exp_dir)
        right_set = set(right)
        left = LayoutGeneratorUtils.getLayoutHugepages(left_layout, self.exp_dir)
        left_set = set(left)

        pebs_set = set(self.pebs_df['PAGE_NUMBER'].to_list())
        all_set = left_set | right_set | pebs_set
        all = list(all_set)

        union_set = left_set | right_set
        union = list(union_set)
        intersection = list(left_set & right_set)
        only_in_left = list(left_set - right_set)
        only_in_right = list(right_set - left_set)
        not_in_right = list(all_set - right_set)

        #assert (len(only_in_left) == 0 and len(only_in_right) > 0), f'Unexpected behavior: the left layout ({left["layout"]}) is included in the right layout ({right["layout"]})'
        #print('******************************************')

        not_in_pebs = list(all_set - pebs_set)
        out_union_based_on_pebs = list(pebs_set - union_set)
        out_union = list(all_set - union_set)

        return only_in_right, only_in_left, out_union, all

    def addPagesFromWorkingSet(self, base_pages, working_set, desired_pebs_coverage, tail=True, epsilon=0.5):
        """Greedily adds pages from `working_set` to `base_pages` to reach a target PEBS coverage.

        Pages are added in order of increasing (`tail=True`) or decreasing
        (`tail=False`) individual TLB coverage until the cumulative coverage
        lands within `[desired_pebs_coverage, desired_pebs_coverage + epsilon]`.

        Args:
            base_pages: Starting page list to add to (not mutated).
            working_set: Candidate pages to add from (pages already in
                `base_pages` are excluded).
            desired_pebs_coverage: Target cumulative PEBS coverage; must be
                >= the coverage of `base_pages` alone.
            tail: If True, prefers adding many small-coverage "tail" pages;
                if False, prefers fewer large-coverage "head" pages.
            epsilon: Acceptable overshoot above `desired_pebs_coverage`.

        Returns:
            tuple[list[int] | None, float]: The augmented page list and its
            coverage, or ``(None, 0)`` if no matching subset was found.
        """
        base_pages_pebs = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, base_pages)

        if desired_pebs_coverage < base_pages_pebs:
            return None, 0

        working_set_df = self.pebs_df.query(f'PAGE_NUMBER in {working_set} and PAGE_NUMBER not in {base_pages}')
        if len(working_set_df) == 0:
            print(f'[DEBUG]: there is no more pages in pebs that can be added')
            return None, 0

        candidate_pebs_coverage = working_set_df['TLB_COVERAGE'].sum()
        #print(f'[DEBUG]: trying to add pages to ({len(base_pages)} pages) from a working-set of {len(working_set)} pages')
        #print(f'[DEBUG]: working-set length after filtering out base pages is {len(working_set_df)} pages')
        #print(f'[DEBUG]: working-set total coverage: {candidate_pebs_coverage} and desired coverage is: {desired_pebs_coverage:.3f}')

        tail_head_order='tail' if tail else 'head'
        #print(f'[DEBUG]: addPagesFromWorkingSet: trying to add {tail_head_order} pages to get a coverage of : {desired_pebs_coverage:.3f}')

        if candidate_pebs_coverage + base_pages_pebs < desired_pebs_coverage:
            #print('[DEBUG]: maximal pebs coverage using working-set is less than desired pebs coverage')
            return None, 0

        df = working_set_df.sort_values('TLB_COVERAGE', ascending=tail)

        added_pages = []
        min_pebs_coverage = desired_pebs_coverage
        max_pebs_coverage = desired_pebs_coverage + epsilon
        total_weight = base_pages_pebs
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            updated_total_weight = total_weight + weight
            if updated_total_weight < max_pebs_coverage:
                added_pages.append(page)
                total_weight = updated_total_weight
            if max_pebs_coverage >= total_weight >= min_pebs_coverage:
                break
        if len(added_pages) == 0:
            return None, 0
        new_pages = base_pages + added_pages
        new_pages.sort()
        new_pebs_coverage = self.pebs_df.query(f'PAGE_NUMBER in {new_pages}')['TLB_COVERAGE'].sum()

        if max_pebs_coverage < new_pebs_coverage or new_pebs_coverage < min_pebs_coverage:
            #print(f'Could not find pages subset with a coverage of {desired_pebs_coverage}')
            #print(f'\t pages subset that was found has:')
            #print(f'\t\t added pages: {len(added_pages)} to {len(base_pages)} pages of the base layout')
            #print(f'\t\t pebs coverage: {new_pebs_coverage}')
            return None, 0

        print(f'Found pages subset with a coverage of {desired_pebs_coverage}')
        print(f'\t pages subset that was found has:')
        print(f'\t\t added pages: {len(added_pages)} to {len(base_pages)} pages of the base layout ==> total pages: {len(new_pages)}')
        print(f'\t\t pebs coverage: {new_pebs_coverage}')
        return new_pages, new_pebs_coverage

    def removePagesByFactor(self, right, left, factor):
        """Removes ``1/factor`` of the pages unique to `left` (relative to `right`).

        Args:
            right: Name of the layout to converge towards (its pages are
                always kept).
            left: Name of the layout to remove pages from.
            factor: Keep every ``factor``-th unique page removed (must be
                >= 2); larger factors remove fewer pages.

        Returns:
            tuple[list[int], float]: The reduced page set and its PEBS
            coverage.
        """
        factor = int(factor)
        print(f'[DEBUG]: removing pages from {left} to get close to {right} by a factor: {factor}')
        assert factor >= 2

        left_pages = LayoutGeneratorUtils.getLayoutHugepages(left, self.exp_dir)
        right_pages = LayoutGeneratorUtils.getLayoutHugepages(right, self.exp_dir)

        candidate_pages = list(set(left_pages) - set(right_pages))
        candidate_pages.sort()
        remove_pages = candidate_pages[::factor]
        for p in remove_pages:
            candidate_pages.remove(p)
        new_pages = right_pages + candidate_pages
        #new_pages = list(set(left_pages) - set(remove_pages))
        new_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, new_pages)
        return new_pages, new_coverage

    def removePagesRecursively(self):
        """Blindly halves the page-removal factor repeatedly to bisect the current max gap.

        Adapts the removal factor based on whether the previous attempt (via
        this same strategy) landed above or below its target, then delegates
        to `removePagesByFactor`. Used as a final fallback inside
        `autoReduceMaximalGap` when factor- and coverage-based mixing both
        fail.

        Returns:
            tuple[str, str, float, list[int], float]: ``(base_layout,
            inc_base, factor, new_pages, new_coverage)``.
        """
        right, left = self.state_log.getMaxGapLayouts()

        last_layout = self.state_log.getLastLayoutName()
        last_inc_base = self.state_log.getIncBaseLayout(last_layout)
        last_base = self.state_log.getBaseLayout(last_layout)
        last_direction = self.state_log.getLayoutScanDirection(last_layout)
        last_real_coverage = self.state_log.getRealCoverage(last_layout)
        last_factor = self.state_log.getLayoutScanValue(last_layout)
        last_inc_base_real = self.state_log.getRealCoverage(last_inc_base)
        last_base_real = self.state_log.getRealCoverage(last_base)
        # check if last scan was done using this method and worked out
        if last_direction != 'remove':
            left = self.state_log.getLeftLayoutName()
            factor = 2
        elif last_inc_base_real < last_real_coverage < last_base_real:
            factor = 2
        elif last_real_coverage <= last_inc_base_real:
            left = last_base
            right = last_inc_base
            factor = last_factor * 2
        elif last_real_coverage >= last_base_real:
            left = last_layout
            right = last_inc_base
            factor = 2

        base_layout = left
        inc_base = right

        new_pages, new_coverage = self.removePagesByFactor(inc_base, base_layout, factor)
        return base_layout, inc_base, factor, new_pages, new_coverage

    def addPagesByFactor(self, left, right, factor):
        """Adds ``(factor-1)/factor`` of the pages unique to `left` (relative to `right`) into `right`.

        Args:
            left: Name of the layout to take additional pages from.
            right: Name of the base layout to add pages to.
            factor: Keep (add) all but every ``factor``-th unique page (must
                be >= 2); larger factors add more pages.

        Returns:
            tuple[list[int], float]: The augmented page set and its PEBS
            coverage.
        """
        factor = int(factor)
        assert factor >= 2

        left_pages = LayoutGeneratorUtils.getLayoutHugepages(left, self.exp_dir)
        right_pages = LayoutGeneratorUtils.getLayoutHugepages(right, self.exp_dir)

        candidate_pages = list(set(left_pages) - set(right_pages))
        candidate_pages.sort()
        removed_pages = candidate_pages[::factor]
        added_pages = list(set(candidate_pages) - set(removed_pages))
        new_pages = right_pages + added_pages
        new_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, new_pages)

        print('[DEBUG]: addPagesByFactor:')
        print(f'\t added {len(added_pages)} pages from {left} to {right}')
        print(f'\t using 1/{factor} of {len(candidate_pages)} distinct {left} pages')
        print(f'\t new total pages: {len(new_pages)}')
        print(f'\t new pebs coverage: {new_coverage}')

        return new_pages, new_coverage

    def addPagesFromLeftLayout(self):
        """Blindly adds pages from the left (higher-coverage) bound of the current max gap.

        Picks an increasing factor for `addPagesByFactor` if the same
        base/increment pair was already tried; if the resulting page set is
        empty or a duplicate, falls back to
        `removePagesBasedOnRealCoverage` targeting the gap's real-coverage
        midpoint.

        Returns:
            tuple[list[int] | None, float, str, str, float]: ``(pages,
            pebs_coverage, base_layout, inc_layout, factor)``.
        """
        right, left = self.state_log.getMaxGapLayouts(False)
        print(f'[DEBUG]: addPagesFromLeftLayout: trying to close max gap between {right} and {left} by adding pages from {left} to {right} blindly')

        base_layout = left
        inc_layout = right
        last_layout = self.state_log.getLastLayoutName()
        last_base = self.state_log.getBaseLayout(last_layout)
        last_inc = self.state_log.getIncBaseLayout(last_layout)
        last_factor = self.state_log.getLayoutScanValue(last_layout)

        factor = 2
        if base_layout == last_base and inc_layout == last_inc:
            factor = last_factor + 1
            factor = max(factor, 2)

        pages, pebs_coverage = self.addPagesByFactor(left, right, factor)
        if pages is None or self.pagesSetExist(pages):
            expected_real_coverage = (self.state_log.getRealCoverage(right) + self.state_log.getRealCoverage(left)) / 2
            pages, pebs_coverage = self.removePagesBasedOnRealCoverage(left, expected_real_coverage)

        return pages, pebs_coverage, base_layout, inc_layout, factor

    def addPagesToBasePages(self, base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage, tail=True):
        """
        Add pages to base_layout_pages to get a total pebs-coverage as close as
        possible to desired_pebs_coverage. The pages should be added from
        add_working_set. If cannot find pages subset from add_working_set
        that covers desired_pebs_coverage, then try to remove from the
        remove_working_set and retry finding a new pages subset.
        """
        if len(add_working_set) == 0:
            return None, 0

        if remove_working_set is None:
            remove_working_set = []

        # make sure that remove_working_set is a subset of the base-layout pages
        assert len( set(remove_working_set) - set(base_layout_pages) ) == 0

        # sort remove_working_set pages by coverage ascendingly
        remove_pages_subset = self.pebs_df.query(f'PAGE_NUMBER in {remove_working_set}').sort_values('TLB_COVERAGE')['PAGE_NUMBER'].to_list()
        not_in_pebs = list(set(remove_working_set) - set(remove_pages_subset))
        remove_pages_subset += not_in_pebs

        i = 0
        pages = None
        max_threshold = 0.5
        while pages is None or self.pagesSetExist(pages):
            threshold = 0.1
            while pages is None and threshold <= max_threshold:
                pages, pebs_coverage = self.addPagesFromWorkingSet(base_layout_pages, add_working_set, desired_pebs_coverage, tail, threshold)
                threshold += 0.1
            # if cannot find pages subset with the expected coverage
            # then remove the page with least coverage and try again
            if i >= len(remove_pages_subset):
                break
            base_layout_pages.remove(remove_pages_subset[i])
            i += 1

        if pages is None or self.pagesSetExist(pages):
            return None, 0

        print(f'[DEBUG] - addPagesToBasePages:')
        print(f'\t layout has {len(base_layout_pages)} pages')
        print(f'\t the new layout has {len(pages)} pages with pebs-coverage: {pebs_coverage}')
        num_common_pages = len( set(pages) & set(base_layout_pages) )
        num_added_pages = len(pages) - num_common_pages
        num_removed_pages = len(base_layout_pages) - num_common_pages
        print(f'\t {num_added_pages} pages were added')
        print(f'\t {num_removed_pages} pages were removed')

        return pages, pebs_coverage

    def getHeadPages(self, num_pages, desired_pebs_coverage):
        """Selects up to `num_pages` of the heaviest PEBS pages without exceeding `desired_pebs_coverage`.

        Args:
            num_pages: Maximum number of pages to select.
            desired_pebs_coverage: Coverage budget; each selected page's
                coverage is subtracted from it.

        Returns:
            list[int]: The selected page numbers, heaviest first.
        """
        pages = []
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        for index, row in df.iterrows():
            if num_pages == 0 or desired_pebs_coverage == 0:
                break
            page = row['PAGE_NUMBER']
            coverage = row['TLB_COVERAGE']
            if coverage <= desired_pebs_coverage:
                desired_pebs_coverage -= coverage
                pages.append(page)
                num_pages -= 1
        return pages

    def addTailPages(self, base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage):
        """Convenience wrapper: `addPagesToBasePages` preferring many small ("tail") pages."""
        return self.addPagesToBasePages(base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage, True)

    def addMinimalHeadPagesByWeight(self, base_layout_pages, add_working_set, head_pages_working_set, desired_pebs_coverage, add_from_tail=True):
        """Searches subsets of "heavy" pages (>= `HEAD_PAGES_WEIGHT_THRESHOLD`) to combine with tail-page filling.

        Enumerates every subset (by increasing size) of head pages found in
        `head_pages_working_set`/`base_layout_pages`, and for each subset
        that doesn't already exceed `desired_pebs_coverage`, tries to fill
        the remainder with `add_working_set` pages via `addPagesToBasePages`.
        Returns as soon as a novel, not-yet-tried layout is found. This
        combinatorial search is needed because a single heavy page can
        overshoot the desired coverage if added greedily.

        Args:
            base_layout_pages: Starting page list.
            add_working_set: Candidate pages available to fill with (tail
                pages, after head pages are excluded).
            head_pages_working_set: Candidate heavy pages to combine
                combinatorially.
            desired_pebs_coverage: Target cumulative PEBS coverage.
            add_from_tail: Whether the fill step should prefer light or
                heavy pages.

        Returns:
            tuple[list[int] | None, float]: The resulting page set and PEBS
            coverage, or ``(None, -1)`` if no combination worked.
        """
        if head_pages_working_set is None:
            head_pages_working_set = []
        head_pages_df = self.pebs_df.query(f'PAGE_NUMBER in {head_pages_working_set} or PAGE_NUMBER in {base_layout_pages}')
        head_pages_df = head_pages_df.query(f'TLB_COVERAGE >= {HEAD_PAGES_WEIGHT_THRESHOLD}')
        head_pages = head_pages_df.sort_values('TLB_COVERAGE', ascending=True)['PAGE_NUMBER'].to_list()
        head_pages_num = len(head_pages)
        head_pages_group = [[head_pages[i]] for i in range(head_pages_num)]

        # filter-out head pages from the base layout to allow adding them gradually
        add_working_set = list( set(add_working_set) - set(head_pages) )

        head_pages_dict = dict()
        for subset_size in range(head_pages_num + 1):
            for subset in itertools.combinations(head_pages_group, subset_size):
                head_pages = list(itertools.chain(*subset))
                head_pages_pebs = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, head_pages)
                head_pages_dict[tuple(head_pages)] = head_pages_pebs
        head_pages_dict = sorted(head_pages_dict.items(), key=lambda kv: kv[1])
        head_pages_dict = collections.OrderedDict(head_pages_dict)

        for head_pages, pebs in head_pages_dict.items():
            # add the head-pages susbet to the base-layout pages for
            # considering them when adding tail pages
            new_base_layout_pages = base_layout_pages + list(head_pages)
            new_base_pages_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, new_base_layout_pages)
            if new_base_pages_coverage > desired_pebs_coverage:
                continue

            pages, pebs = self.addPagesToBasePages(new_base_layout_pages, add_working_set, [], desired_pebs_coverage, add_from_tail)
            if pages is not None and not self.pagesSetExist(pages):
                return pages, pebs

        return None, -1

    def addMinimalHeadPagesByNumber(self, base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage, add_from_tail=True):
        """Like `addMinimalHeadPagesByWeight`, but combinatorially searches by page count rather than by weight threshold.

        Considers only heavy (>= 5.0 coverage) candidate pages from
        `add_working_set` not already in `base_layout_pages`, enumerating
        subsets by increasing size and, for each, attempting to fill the
        rest via `addTailPages`/`addPagesToBasePages`.

        Returns:
            tuple[list[int] | None, float]: The resulting page set and PEBS
            coverage, or ``(None, -1)`` if no combination worked.
        """
        head_pages_df = self.pebs_df.query(f'PAGE_NUMBER in {add_working_set} and PAGE_NUMBER not in {base_layout_pages} and TLB_COVERAGE >= 5.0')
        head_pages = head_pages_df.sort_values('TLB_COVERAGE')['PAGE_NUMBER'].to_list()
        head_pages_num = len(head_pages)
        head_pages_list = [[head_pages[i]] for i in range(head_pages_num)]

        # filter out head pages from the working-set
        filtered_working_set = list( set(add_working_set) - set(head_pages) )

        for subset_size in range(head_pages_num + 1):
            for subset in itertools.combinations(head_pages_list, subset_size):
                head_pages_subset = list(itertools.chain(*subset))
                # work on a copy of the base_layout_pages
                new_base_layout_pages = base_layout_pages.copy()
                # add the head-pages susbet to the base-layout pages for
                # considering them when adding tail pages
                new_base_layout_pages += head_pages_subset
                new_base_pages_coverage = LayoutGeneratorUtils.calculateTlbCoverage(self.pebs_df, new_base_layout_pages)
                if new_base_pages_coverage > desired_pebs_coverage:
                    continue

                if add_from_tail:
                    # try to add tail pages
                    pages, pebs = self.addTailPages(new_base_layout_pages, filtered_working_set, remove_working_set, desired_pebs_coverage)
                else:
                    # try to add head pages
                    pages, pebs = self.addPagesToBasePages(new_base_layout_pages, filtered_working_set, remove_working_set, desired_pebs_coverage, False)

                if pages is not None and not self.pagesSetExist(pages):
                    return pages, pebs

        return None, -1

    def addMinimalHeadPages(self, base_layout_pages, add_working_set, head_pages_working_set, desired_pebs_coverage, add_from_tail=True):
        """Dispatches to the weight-based head-page combination search (`addMinimalHeadPagesByWeight`)."""
        return self.addMinimalHeadPagesByWeight(base_layout_pages, add_working_set, head_pages_working_set, desired_pebs_coverage, add_from_tail)

    def addHeadPages(self, base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage):
        """Adds pages preferring few, heavy ("head") pages, trying head-first then tail-first filling."""
        #return self.addPagesToBasePages(base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage, False)
        pages, pebs = self.addMinimalHeadPages(base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage, False)
        if pages is None:
            pages, pebs = self.addMinimalHeadPages(base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage, True)
        return pages, pebs

    def addPages(self, base_layout, add_working_set, remove_working_set, desired_pebs_coverage, tail=True):
        """Adds pages to `base_layout` targeting `desired_pebs_coverage`, choosing a tail- or head-first strategy.

        Args:
            base_layout: Name of the layout to add pages to.
            add_working_set: Candidate pages to add from.
            remove_working_set: Candidate pages to remove if adding alone
                cannot reach the target (passed through to
                `addPagesToBasePages`).
            desired_pebs_coverage: Target cumulative PEBS coverage.
            tail: If True, use `addTailPages`; if False, use `addHeadPages`.

        Returns:
            tuple[list[int] | None, float]: The resulting page set and PEBS
            coverage.
        """
        base_layout_pages = LayoutGeneratorUtils.getLayoutHugepages(base_layout, self.exp_dir)
        if tail:
            return self.addTailPages(base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage)
        return self.addHeadPages(base_layout_pages, add_working_set, remove_working_set, desired_pebs_coverage)

    def removePagesBasedOnRealCoverage(self, base_layout, expected_real_coverage):
        """Removes pages from `base_layout` targeting a *real*-coverage goal, via a linear PEBS-to-real scale.

        Scales `expected_real_coverage` into an equivalent PEBS-coverage
        target using `base_layout`'s own real-to-PEBS ratio, then delegates
        to `removePages`.

        Args:
            base_layout: Name of the layout to remove pages from.
            expected_real_coverage: Desired real-coverage percentage.

        Returns:
            tuple[list[int] | None, float]: The reduced page set and PEBS
            coverage.
        """
        base_layout_real_coverage = self.state_log.getRealCoverage(base_layout)
        base_layout_pebs_coverage = self.state_log.getPebsCoverage(base_layout)
        base_layout_real_to_pebs_scale = base_layout_pebs_coverage / base_layout_real_coverage
        scaled_desired_coverage = base_layout_real_to_pebs_scale * expected_real_coverage

        print(f'[DEBUG]: desired real coverage: {expected_real_coverage}')
        print(f'[DEBUG]: scaled desired pebs coverage: {scaled_desired_coverage}')

        return self.removePages(base_layout, None, scaled_desired_coverage)

    def removePages(self, base_layout, working_set, desired_pebs_coverage, tail=True):
        """Removes pages from `base_layout` to reach `desired_pebs_coverage`, retrying over the full layout if `working_set` fails.

        Args:
            base_layout: Name of the layout to remove pages from.
            working_set: Restricted candidate pages to remove from, or
                ``None`` to consider all of `base_layout`'s pages.
            desired_pebs_coverage: Target cumulative PEBS coverage.
            tail: If True, remove light pages first; if False, remove heavy
                pages first.

        Returns:
            tuple[list[int] | None, float]: The reduced page set and PEBS
            coverage, or ``(None, 0)`` if no valid subset was found.
        """
        pages, pebs = self.removePagesInOrder(base_layout, working_set, desired_pebs_coverage, tail)
        if pages is None or self.pagesSetExist(pages):
            pages, pebs = self.removePagesInOrder(base_layout, None, desired_pebs_coverage, tail)
        if pages is None or self.pagesSetExist(pages):
            return None, 0
        return pages, pebs

    def removePagesInOrder(self, base_layout, working_set, desired_pebs_coverage, tail=True):
        """Removes pages from `working_set` in coverage order until reaching `desired_pebs_coverage`.

        Args:
            base_layout: Name of the layout to remove pages from.
            working_set: Candidate pages to remove, or ``None`` to use all
                of `base_layout`'s pages.
            desired_pebs_coverage: Target cumulative PEBS coverage (within
                0.2 points).
            tail: If True, removes lightest pages first; if False, removes
                heaviest pages first.

        Returns:
            tuple[list[int] | None, float]: The reduced page set and its
            PEBS coverage, or ``(None, 0)`` if no pages could be removed.
        """
        base_layout_pages = LayoutGeneratorUtils.getLayoutHugepages(base_layout, self.exp_dir)
        base_layout_coverage = self.state_log.getPebsCoverage(base_layout)
        if working_set is None:
            working_set = base_layout_pages
        df = self.pebs_df.query(f'PAGE_NUMBER in {working_set}')
        df = df.sort_values('TLB_COVERAGE', ascending=tail)
        print(f'[DEBUG]: removePages: {base_layout} has {len(base_layout_pages)} total pages, and {len(df)} pages in pebs as candidates to be removed')

        removed_pages = []
        total_weight = base_layout_coverage
        epsilon = 0.2
        max_coverage = desired_pebs_coverage
        min_coverage = desired_pebs_coverage - epsilon
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            updated_total_weight = total_weight - weight
            if updated_total_weight > min_coverage:
                removed_pages.append(page)
                total_weight = updated_total_weight
            if max_coverage >= total_weight >= min_coverage:
                break
        if len(removed_pages) == 0:
            return None, 0
        new_pages = list(set(base_layout_pages) - set(removed_pages))
        new_pages.sort()
        new_pebs_coverage = self.pebs_df.query(f'PAGE_NUMBER in {new_pages}')['TLB_COVERAGE'].sum()

        print(f'[DEBUG]: total removed pages from {base_layout}: {len(removed_pages)}')
        print(f'[DEBUG]: new layout coverage: {new_pebs_coverage}')

        return new_pages, new_pebs_coverage

    def realToPebsCoverageBasedOnExistingLayout(self, layout, expected_real_coverage, scan_direction, scan_order):
        """
        1) find the real-coverage to expected-real-coverage ratio of layout
        2) scale layout pebs based on this ratio (1) (i.e., what is the pebs
        value that if will be used then expected-real-coverage will be obtained)
        3) find the predcited-pebs (2) to real coverage ratio
        4) scale the expected_real_coverage based on the ratio calculated in (3)

        For head-order scans, coverage tracks 1:1 with real progress instead
        (see the ``'head' in scan_order`` special case below), so no ratio
        scaling is needed there.

        Args:
            layout: Name of the reference layout whose PEBS/real-coverage
                relationship is used to build the scaling ratio.
            expected_real_coverage: The target real-coverage percentage to
                translate into a PEBS-coverage target.
            scan_direction: ``'add'`` or ``'remove'`` (unused directly here,
                kept for API symmetry with sibling methods).
            scan_order: ``'tail'`` or ``'head'``; determines which scaling
                formula is used.

        Returns:
            float: The predicted PEBS-coverage value expected to yield
            `expected_real_coverage` once measured.
        """
        layout_real = self.state_log.getRealCoverage(layout)
        layout_pebs = self.state_log.getPebsCoverage(layout)
        layout_expected_real = self.state_log.getExpectedRealCoverage(layout)

        if 'head' in scan_order:
            scaled_desired_coverage = layout_expected_real - layout_real + layout_pebs
            return scaled_desired_coverage

        # prevent division by zero and getting numerous ratio in
        # the calculation of expected_to_real
        layout_real = max(1, layout_real)
        expected_to_real = layout_expected_real / layout_real
        scaled_pebs = layout_pebs * expected_to_real
        scaled_pebs_to_real = scaled_pebs / layout_expected_real
        scaled_desired_coverage = scaled_pebs_to_real * expected_real_coverage

        return scaled_desired_coverage

    def scaleLastLayoutToExpectedCoverage(self, expected_real_coverage):
        """Quick-path prediction: extrapolates the last 'add' layout's pebs-vs-real trend.

        If the last created layout used the ``'add'`` scan direction and is
        still relevant (its increment base still matches the current
        target) but undershot its own expected real coverage (`real_gap` <=
        0), doubles its PEBS-coverage delta from its base layout as the next
        target, as long as that stays under 100%. This lets the algorithm
        make faster progress when a previous step's coverage prediction was
        too conservative.

        Args:
            expected_real_coverage: The real-coverage value being targeted.

        Returns:
            tuple[float | None, str | None]: ``(desired_pebs_coverage,
            base_layout)`` if this fast path applies, otherwise ``(None,
            None)`` so the caller falls back to `tryToConcludeNextCoverage`.
        """
        last_layout = self.state_log.getLastLayoutName()
        if self.state_log.getLayoutScanDirection(last_layout) == 'remove':
            return None, None

        last_pebs = self.state_log.getPebsCoverage(last_layout)
        last_real = self.state_log.getRealCoverage(last_layout)

        pebs_delta = self.state_log.getPebsCoverageDeltaBetweenLayoutAndItsBase(last_layout)
        real_gap = self.state_log.getGapBetweenLayoutAndItsBase(last_layout)
        if pebs_delta is None or real_gap is None:
            return None, None

        # if the increment base-layout was changed, then fallback
        if self.state_log.getIncBaseLayout(last_layout) != self.state_log.getNextIncrementBase():
            return None, None

        if real_gap <= 0:
            desired_coverage = last_pebs + pebs_delta * 2
            base_layout = last_layout
            if desired_coverage < 100:
                return desired_coverage, base_layout

        return None, None


    def tryToConcludeNextCoverage(self, base_layout, expected_real_coverage, scan_direction, scan_order):
        """Predicts the PEBS-coverage target that should yield `expected_real_coverage`.

        First tries the fast path (`scaleLastLayoutToExpectedCoverage`).
        Failing that, looks among previously measured layouts sharing the
        same `scan_direction`/`scan_order` for the pair that most tightly
        brackets `expected_real_coverage`, and linearly interpolates (or
        extrapolates, if only one side is available) their PEBS-coverage
        values via `realToPebsCoverageBasedOnExistingLayout`. This is the
        main heuristic used to decide how many additional PEBS-coverage
        points a new candidate layout should target before it is generated
        and actually measured.

        Args:
            base_layout: Fallback base layout name, used if no better
                candidate is found.
            expected_real_coverage: Target real-coverage percentage.
            scan_direction: ``'add'`` or ``'remove'``.
            scan_order: ``'tail'`` or ``'head'``.

        Returns:
            tuple[float | None, str]: ``(desired_pebs_coverage,
            base_layout)`` -- `desired_pebs_coverage` is ``None`` if no
            prediction could be made (caller should fall back to a
            heuristic default).
        """
        desired_coverage, new_base_layout = self.scaleLastLayoutToExpectedCoverage(expected_real_coverage)
        if desired_coverage is not None:
            return desired_coverage, new_base_layout

        base_layout_pages = LayoutGeneratorUtils.getLayoutHugepages(base_layout, self.exp_dir)
        selected_layouts = []

        # get all layouts that have the same scan direction (add/remove)
        #query = self.state_log.df.query(f'scan_direction == "{scan_direction}" and scan_order == "{scan_order}"')
        query_str = f'scan_direction == "{scan_direction}" and scan_order == "{scan_order}"'
        #query = self.state_log.df.query(f'({query_str}) or (scan_direction == "auto")')
        query = self.state_log.df.query(f'{query_str}')
        if len(query) == 0:
            return None, base_layout
        if len(query) == 1:
            # if there is only one layout with the same required direction and
            # order then try to predict the next coverage by scaling the found
            # layout pebs value based on its expected vs real coverage
            layout = query.iloc[0]['layout']
            desired_coverage = self.realToPebsCoverageBasedOnExistingLayout(layout, expected_real_coverage, scan_direction, scan_order)
            if scan_direction == 'add' and self.state_log.getRealCoverage(layout) < expected_real_coverage:
                base_layout = layout
            elif scan_direction == 'remove' and self.state_log.getRealCoverage(layout) > expected_real_coverage:
                base_layout = layout
            return desired_coverage, base_layout

        if scan_direction == 'add':
            for l in query['layout']:
                pages = LayoutGeneratorUtils.getLayoutHugepages(l, self.exp_dir)
                # check if one pages set is included in the other
                common_pages = set(pages) & set(base_layout_pages)
                if common_pages == set(pages) or common_pages == set(base_layout_pages):
                    selected_layouts.append(l)
            # add the right/left layouts if the current scan range
            selected_layouts.append(self.state_log.getRightLayoutName())
        else:
            # when removing consider all relevant layouts
            selected_layouts = query['layout'].to_list()

        # keep only the previous selected layouts
        query = self.state_log.df.query(f'layout in {selected_layouts}')
        if len(query) == 0:
            return None, base_layout

        # select all layouts that are in the right side if the desired coverage
        # and then select the one with the maximal pebs coverage
        right_layouts = query.query(f'real_coverage < {expected_real_coverage}').sort_values('pebs_coverage')
        if len(right_layouts) > 0:
            right = right_layouts.iloc[-1]
            right_layout = right['layout']
            right_pebs = self.state_log.getPebsCoverage(right_layout)
            right_real = self.state_log.getRealCoverage(right_layout)
        else:
            right = right_layout = None
            right_pebs = 0

        # select all layouts that are in the left side if the desired coverage
        # with a pebs coverage greater than the selected right layout
        # and then select from them the layout with the least pebs coverage
        left_layouts = query.query(f'real_coverage > {expected_real_coverage} and pebs_coverage > {right_pebs}').sort_values('pebs_coverage')
        if len(left_layouts) > 0:
            left = left_layouts.iloc[0]
            left_layout = left['layout']
            left_pebs = self.state_log.getPebsCoverage(left_layout)
            left_real = self.state_log.getRealCoverage(left_layout)
        else:
            left = left_layout = None

        if right is None and left is None:
            print('[DEBUG]: tryToConcludeNextCoverage - could not find layouts to use for the prediction')
            return None, base_layout

        print(f'[DEBUG]: tryToConcludeNextCoverage - the surrounding layouts:  {right_layout} < {expected_real_coverage} < {left_layout}')

        if right is None:
            # left is not None
            desired_coverage = self.realToPebsCoverageBasedOnExistingLayout(left_layout, expected_real_coverage, scan_direction, scan_order)
            print(f'[DEBUG]: predicting next pebs coverage based on {left_layout} left-layout to: {desired_coverage}')
            if scan_direction == 'remove':
                base_layout = left_layout
            return desired_coverage, base_layout

        if left is None:
            # right is not None
            desired_coverage = self.realToPebsCoverageBasedOnExistingLayout(right_layout, expected_real_coverage, scan_direction, scan_order)
            print(f'[DEBUG]: predicting next pebs coverage based on {right_layout} right-layout to: {desired_coverage}')
            if scan_direction == 'add':
                base_layout = right_layout
            return desired_coverage, right_layout

        # scale based on the lower pebs coverage
        scaled_right_pebs_coverage = self.realToPebsCoverageBasedOnExistingLayout(right_layout, expected_real_coverage, scan_direction, scan_order)
        scaled_left_pebs_coverage = self.realToPebsCoverageBasedOnExistingLayout(left_layout, expected_real_coverage, scan_direction, scan_order)

        # prefer scaling by the lower pebs-coverage, which is of the right
        # layout. If the right layout scaled pebs-coverage falls out the
        # right-left layouts range then consider the left layout pebs-coverage,
        # and if it's outside the range then consider the average as the
        # desired-coverage candidate
        if left_pebs < scaled_right_pebs_coverage < right_pebs:
            desired_coverage = scaled_right_pebs_coverage
        elif left_pebs < scaled_left_pebs_coverage < right_pebs:
            # if the left layout is closed to the expected-real-coverage then scale based on it
            desired_coverage = scaled_left_pebs_coverage
        else:
            # if the scaled pebs coverage falls outside the range between right and left
            # then consider desired-coverage as the average of the right and left pebs values
            pebs_avg = (right_pebs + left_pebs) / 2
            desired_coverage = pebs_avg

        print(f'[DEBUG]: predicting next pebs coverage based on {right_layout} and {left_layout} to: {desired_coverage}')
        if scan_direction == 'add':
            base_layout = right_layout
        elif scan_direction == 'remove':
            base_layout = left_layout
        return desired_coverage, base_layout

    def getAllLayoutsFromStateLogs(self):
        """Loads/initializes a `StateLog` for every anchor subgroup and collects all known layout names.

        Called once at construction time so that `pagesSetExist` can check
        newly proposed page sets against every layout ever created, not just
        ones in the currently active subgroup.

        Returns:
            list[str]: The names of every layout referenced across all
            subgroup `StateLog`\\ s.
        """
        layouts = []
        for i in range(len(self.subgroups_log.df)-1):
            right, left = self.subgroups_log.getSubgroup(i)
            right_layout = right['layout']
            left_layout = left['layout']
            # initialize the state-log for the current group
            self.state_log = StateLog(
                    self.exp_dir,
                    self.results_df,
                    right_layout,
                    left_layout,
                    self.max_gap,
                    self.max_budget,
                    self.debug)
            # if the state log is empty then it seems just now we are
            # about to start scanning this group
            self.updateStateLog(right, left)
            layouts += self.state_log.getAllLayouts()
        layouts = list(set(layouts))
        return layouts

    def pagesSetExist(self, pages_to_find):
        """Checks whether `pages_to_find` is identical to the page set of any previously created layout.

        Used throughout the page-selection strategies to reject candidate
        page sets that would duplicate an already-measured layout.

        Args:
            pages_to_find: Candidate page set to check.

        Returns:
            bool: True if an identical layout already exists.
        """
        for layout in self.all_layouts:
            pages = LayoutGeneratorUtils.getLayoutHugepages(layout, self.exp_dir)
            if set(pages) == set(pages_to_find):
                print(f'===== found identical layout: {layout} =====')
                return True
        return False

    def updateAddScanParametersCornerCase(self, scan_direction, scan_order, desired_pebs_coverage):
        """Handles the edge case where an 'add' scan's target coverage would exceed 100%.

        If the target PEBS coverage is essentially 100% but the left
        (upper) layout bound is not yet the all-hugepages layout, backs off
        the target to a smaller intermediate value. If the left bound
        *already* is (approximately) the all-hugepages layout, there are no
        more PEBS-sampled pages to add, so falls back to a blind
        (``'auto'``/``'blind'``) scan instead.

        Args:
            scan_direction: Current scan direction (typically ``'add'``).
            scan_order: Current scan order (``'tail'`` or ``'head'``).
            desired_pebs_coverage: The originally computed target coverage.

        Returns:
            tuple[str, str, float]: The possibly-adjusted
            ``(scan_direction, scan_order, desired_pebs_coverage)``.
        """
        last_layout = self.state_log.getLastLayoutName()
        last_pebs_coverage = self.state_log.getPebsCoverage(last_layout)

        # if the left layout is not the all-2MB layout and we
        # over-estimated desired_pebs_coverage, then fix it
        left_layout = self.state_log.getLeftLayoutName()
        left_pebs_coverage = self.state_log.getPebsCoverage(left_layout)
        if desired_pebs_coverage >= 99.9:
            if left_pebs_coverage >= 99.9:
                # if left layout is the all-2MB layout and we are trying to add
                # more than 100% coverage (i.e., we still need to add more pages
                # to close the real-coverage gap but we have no additional pages
                # in pebs to be added), then add pages blindly, i.e., without
                # considering pebs weights
                scan_direction = 'auto'
                scan_order = 'blind'
            else:
                # update desired_pebs_coverage since we jumped too far
                desired_pebs_coverage = min((last_pebs_coverage + 100) / 2, last_pebs_coverage + self.default_increment)

        return scan_direction, scan_order, desired_pebs_coverage

    def getAddScanParameters(self, base_layout, expected_real_coverage, scan_direction, scan_order):
        """Computes the base layout and target PEBS coverage for an 'add'-direction scan.

        Tries `tryToConcludeNextCoverage` first (optionally retrying with
        the base layout's own base layout as a second attempt); if no
        prediction is available, falls back to scaling either the last
        layout with matching direction/order, or the rightmost/leftmost
        layout (depending on `scan_order`), as the baseline for prediction.
        Finally clamps the result so it never targets less coverage than
        `base_layout` already has.

        Args:
            base_layout: Candidate base layout name to add pages to.
            expected_real_coverage: Target real-coverage percentage.
            scan_direction: Expected to be ``'add'``.
            scan_order: ``'tail'`` or ``'head'``.

        Returns:
            tuple[float, str]: ``(desired_pebs_coverage, base_layout)``.
        """
        predicted_coverage, base_layout = self.tryToConcludeNextCoverage(base_layout, expected_real_coverage, scan_direction, scan_order)
        right_layout = self.state_log.getRightLayoutName()
        if predicted_coverage is None and base_layout != right_layout:
            base_layout = self.state_log.getBaseLayout(base_layout)
            predicted_coverage, base_layout = self.tryToConcludeNextCoverage(base_layout, expected_real_coverage, scan_direction, scan_order)

        if predicted_coverage is None:
            # if cannot predict the next desired_pebs_coveragem then
            # 1) if the last layout has the same scan_order of the current then
            #    use it as a baseline for scaling its coverage to the desired
            #    coverage
            # 2) Otherwise, use the rightmost layout for tail scans and the
            #    leftmost for head scans because the leftmost is mostly has
            #    more head pages than the rightmost and then it's more suitable
            #    for scaling its coverage when using head pages
            last_layout = self.state_log.getLastLayoutName()
            if scan_direction == self.state_log.getLayoutScanDirection(last_layout) \
                    and scan_order == self.state_log.getLayoutScanOrder(last_layout):
                base_layout = self.state_log.getBaseLayout(last_layout)
                desired_pebs_coverage = self.realToPebsCoverageBasedOnExistingLayout(last_layout, expected_real_coverage, scan_direction, scan_order)
            elif scan_order == 'tail':
                right_layout = self.state_log.getRightLayoutName()
                base_layout = right_layout
                desired_pebs_coverage = self.realToPebsCoverageBasedOnExistingLayout(right_layout, expected_real_coverage, scan_direction, scan_order) + self.default_increment
            elif scan_order == 'head':
                right_layout = self.state_log.getRightLayoutName()
                left_layout = self.state_log.getLeftLayoutName()
                base_layout = right_layout
                #desired_pebs_coverage = self.realToPebsCoverageBasedOnExistingLayout(left_layout, expected_real_coverage, scan_direction, scan_order) + self.default_increment
                desired_pebs_coverage = self.state_log.getPebsCoverage(left_layout)
            else:
                assert False,f'unrecognized scan-order={scan_order} for add scan method'
            print(f'[DEBUG]: looking for pebs-coverage: {desired_pebs_coverage} to get real-coverage: {expected_real_coverage}')

        else: # predicted_coverage is not None
            desired_pebs_coverage = predicted_coverage
            print(f'[DEBUG]: predicting next pebs-coverage as {desired_pebs_coverage} to get real-coverage of {expected_real_coverage}')

        base_layout_pebs_coverage = self.state_log.getPebsCoverage(base_layout)
        if desired_pebs_coverage < base_layout_pebs_coverage:
            desired_pebs_coverage = base_layout_pebs_coverage + self.max_gap
        return desired_pebs_coverage, base_layout

    def getRemoveScanParameters(self, base_layout, expected_real_coverage, scan_direction, scan_order):
        """Computes the base layout and target PEBS coverage for a 'remove'-direction scan.

        Mirrors `getAddScanParameters` for the removal case: tries
        `tryToConcludeNextCoverage` (retrying against the left-bound layout
        if needed), then falls back to a linear real-to-PEBS ratio scaling
        from either the last 'remove' layout (if it overshot the target) or
        `base_layout` itself.

        Args:
            base_layout: Candidate base layout name to remove pages from.
            expected_real_coverage: Target real-coverage percentage.
            scan_direction: Expected to be ``'remove'``.
            scan_order: ``'tail'`` or ``'head'``.

        Returns:
            tuple[float, str]: ``(desired_pebs_coverage, base_layout)``.
        """
        predicted_coverage, base_layout = self.tryToConcludeNextCoverage(base_layout, expected_real_coverage, scan_direction, scan_order)
        left_layout = self.state_log.getLeftLayoutName()
        if predicted_coverage is None and base_layout != left_layout:
            base_layout = left_layout
            predicted_coverage, base_layout = self.tryToConcludeNextCoverage(base_layout, expected_real_coverage, scan_direction, scan_order)

        if predicted_coverage is None:
            last_layout = self.state_log.getLastLayoutName()
            last_real = self.state_log.getRealCoverage(last_layout)
            last_scan_direction = self.state_log.getLayoutScanDirection(last_layout)
            if last_scan_direction == 'remove' and last_real > expected_real_coverage:
                last_pebs = self.state_log.getPebsCoverage(last_layout)
                base_layout = last_layout
                desired_pebs_coverage = last_pebs - (last_real - expected_real_coverage)
                if desired_pebs_coverage > 0:
                    return desired_pebs_coverage, base_layout

            pebs_to_real = self.state_log.getPebsCoverage(base_layout) / self.state_log.getRealCoverage(base_layout)
            desired_pebs_coverage = pebs_to_real * expected_real_coverage
            print(f'[DEBUG]: looking for pebs-coverage: {desired_pebs_coverage} to get real-coverage: {expected_real_coverage}')

        else: # predicted_coverage is not None
            desired_pebs_coverage = predicted_coverage
            print(f'[DEBUG]: predicting next pebs-coverage as {desired_pebs_coverage} to get real-coverage of {expected_real_coverage}')

        return desired_pebs_coverage, base_layout

    def getFirstLayoutScanParameters(self, expected_real_coverage, base_layout):
        """Computes initial scan parameters when a subgroup has no prior scans to learn from.

        Currently unused directly by `createLayout` (kept as an explicit,
        documented alternative to `getAddScanParameters` for the
        first-layout-in-subgroup case; see the commented-out call site in
        `getScanParameters`). Estimates the desired PEBS coverage as a
        direct offset from `base_layout`'s own PEBS-vs-real gap.

        Args:
            expected_real_coverage: Target real-coverage percentage.
            base_layout: The layout to scale from.

        Returns:
            tuple[str, str, float]: ``('add', 'tail', desired_pebs_coverage)``.
        """
        base_real_coverage = self.state_log.getRealCoverage(base_layout)
        base_pebs_coverage = self.state_log.getPebsCoverage(base_layout)

        real_range_delta_avg = (self.state_log.getRealCoverage(self.state_log.getLeftLayoutName()) - expected_real_coverage) / 2
        #desired_pebs_coverage =  expected_real_coverage - base_real_coverage + base_pebs_coverage + real_range_delta_avg
        desired_pebs_coverage =  expected_real_coverage - base_real_coverage + base_pebs_coverage
        if desired_pebs_coverage >= 100:
            desired_pebs_coverage = (base_pebs_coverage + 100) / 2

        scan_direction = 'add'
        scan_order = 'tail'

        return scan_direction, scan_order, desired_pebs_coverage

    def getScanOrder(self, default_order):
        """Decides whether to continue with 'tail' or 'head' page scanning based on recent progress.

        If only base (anchor) layouts have been scanned so far, returns
        `default_order` unchanged. Otherwise inspects how much the last
        layout moved in PEBS coverage vs. real coverage relative to its own
        base: if PEBS barely moved but real coverage moved a lot, switches
        to ``'tail'`` (many small pages needed); if real coverage barely
        moved but PEBS moved a lot, switches to ``'head'`` (large pages are
        overshooting); otherwise keeps the last used order.

        Args:
            default_order: Order to use when there is no scan history yet.

        Returns:
            str: ``'tail'`` or ``'head'``.
        """
        if self.state_log.hasOnlyBaseLayouts():
            return default_order

        last_layout = self.state_log.getLastLayoutName()
        last_pebs = self.state_log.getPebsCoverage(last_layout)
        last_real = self.state_log.getRealCoverage(last_layout)

        last_base = self.state_log.getBaseLayout(last_layout)
        base_pebs = self.state_log.getPebsCoverage(last_base)
        base_real = self.state_log.getRealCoverage(last_base)

        pebs_delta = abs(last_pebs - base_pebs)
        real_delta = abs(last_real - base_real)

        if pebs_delta < 1 and real_delta > self.default_increment:
            return 'tail'
        elif real_delta < 1 and pebs_delta > self.default_increment:
            return 'head'
        else:
            return self.state_log.getLayoutScanOrder(last_layout)

    def getScanParameters(self, increment_base, base_layout, expected_real_coverage, scan_direction, scan_order):
        """Dispatches to `getAddScanParameters`/`getRemoveScanParameters` (or falls back to a blind scan).

        Args:
            increment_base: Name of the current increment-base layout
                (unused directly, kept for API symmetry with
                `applyScanParameters`).
            base_layout: Candidate base layout to scan from.
            expected_real_coverage: Target real-coverage percentage.
            scan_direction: ``'add'``, ``'remove'``, or anything else (which
                triggers a blind ``'auto'``/``'blind'`` scan).
            scan_order: ``'tail'`` or ``'head'``.

        Returns:
            tuple[str, str, float | None, str]: ``(scan_direction,
            scan_order, desired_pebs_coverage, base_layout)``.
        """
        if scan_direction == 'add':
            #if self.state_log.hasOnlyBaseLayouts():
            #    scan_direction, scan_order, desired_pebs_coverage = \
            #            self.getFirstLayoutScanParameters(expected_real_coverage, base_layout)
            #else:
            desired_pebs_coverage, base_layout = \
                    self.getAddScanParameters(base_layout, expected_real_coverage, scan_direction, scan_order)
            scan_direction, scan_order, desired_pebs_coverage = \
                self.updateAddScanParametersCornerCase(scan_direction, scan_order, desired_pebs_coverage)
        elif scan_direction == 'remove':
            desired_pebs_coverage, base_layout = \
                    self.getRemoveScanParameters(base_layout, expected_real_coverage, scan_direction, scan_order)
        else:
            scan_direction = 'auto'
            scan_order = 'blind'
            desired_pebs_coverage = None

        return scan_direction, scan_order, desired_pebs_coverage, base_layout

    def applyScanParameters(self, scan_direction, scan_order, \
        desired_pebs_coverage, expected_real_coverage, \
            base_layout, increment_base, \
            main_working_set, secondary_working_set=None):
        """Executes the chosen scan strategy to actually produce a candidate page set.

        Dispatches to `addPages` (``'add'``), `removePages` (``'remove'``,
        always applied against the left/upper-bound layout), or, for the
        blind ``'auto'`` fallback, `addPagesFromLeftLayout` (which may also
        update `base_layout`/`increment_base`/`expected_real_coverage` since
        it does not target a specific coverage value).

        Args:
            scan_direction: ``'add'``, ``'remove'``, or ``'auto'``.
            scan_order: ``'tail'``, ``'head'``, or ``'blind'``.
            desired_pebs_coverage: Target PEBS coverage (ignored for
                ``'auto'``).
            expected_real_coverage: Target real coverage (may be recomputed
                for ``'auto'``).
            base_layout: Layout to scan from.
            increment_base: Current increment-base layout (may be
                recomputed for ``'auto'``).
            main_working_set: Primary candidate page set for this scan.
            secondary_working_set: Fallback candidate page set (used by
                `addPages`'s corner-case handling).

        Returns:
            tuple[list[int] | None, float, str, str, float, float | None]:
            ``(pages, pebs_coverage, base_layout, increment_base,
            expected_real_coverage, factor)`` -- `factor` is only set for
            the ``'auto'`` blind strategy.
        """
        pages = None
        pebs_coverage = -1
        factor = None

        left_layout = self.state_log.getLeftLayoutName()
        tail = (scan_order == 'tail')
        if scan_direction == 'add':
            pages, pebs_coverage = self.addPages(base_layout, main_working_set, secondary_working_set, desired_pebs_coverage, tail)
            if pages is None:
                # if cannot find base pages based on current base layout
                # then fall back to start looking for a new base layout
                # starting from the rightmost layout
                base_layout = self.state_log.getRightLayoutName()
                pages, pebs_coverage = self.addPages(base_layout, main_working_set, secondary_working_set, desired_pebs_coverage, tail)

        if scan_direction == 'remove':
            pages, pebs_coverage = self.removePages(left_layout, main_working_set, desired_pebs_coverage, tail)

        # last chance to find some pages subset
        if scan_direction == 'auto':
            pages, pebs_coverage, base_layout, increment_base, factor = self.addPagesFromLeftLayout()
            expected_real_coverage = (self.state_log.getRealCoverage(base_layout) + self.state_log.getRealCoverage(increment_base)) / 2

        return pages, pebs_coverage, base_layout, increment_base, expected_real_coverage, factor

    def createNextLayoutDynamically(self):
        """Phase-2 entry point: selects a subgroup and tries scan strategies in order until one succeeds.

        Refreshes/selects the active subgroup (`updateLogs`); if no subgroup
        needs work, returns immediately (all gaps already satisfy
        `max_gap`). Otherwise tries, in order: the same scan method as the
        last layout in this subgroup (to keep making consistent progress),
        then ``'add'``, ``'remove'``, ``'add_round2'`` (head-order add),
        ``'auto_reduce-max'`` (bisect the largest gap heuristically), and
        finally ``'auto_blind'`` (fully blind page mixing) --
        see `createLayoutUsingScanMethod`.

        Raises:
            AssertionError: If `results_df` is missing, or if every strategy
                failed to produce a novel layout (should not happen in
                practice).
        """
        assert self.results_df is not None,'results file does not exist'
        # fill or update SubgroupsLog and StateLog
        if not self.updateLogs():
            return
        print('==============================================')
        print(self.state_log.df)
        print('----------------------------------------------')

        done = False
        last_scan_method = self.state_log.getLayoutScanDirection(self.state_log.getLastLayoutName())

        if last_scan_method != 'auto':
            done = done or self.createLayoutUsingScanMethod(last_scan_method)
        done = done or self.createLayoutUsingScanMethod('add')
        done = done or self.createLayoutUsingScanMethod('remove')
        done = done or self.createLayoutUsingScanMethod('add_round2')
        done = done or self.createLayoutUsingScanMethod('auto_reduce-max')
        done = done or self.createLayoutUsingScanMethod('auto_blind')

        assert done, 'cannot create next layout...'

        print('----------------------------------------------')
        print(self.state_log.df)
        print('==============================================')

    def createLayoutUsingScanMethod(self, scan_method='add'):
        """Attempts to create a new layout using one named scan strategy.

        Computes the three working sets relative to the subgroup's current
        right (R) and left (L) bounding layouts:
          - ``alpha`` = pages(R) - pages(L) (right-only)
          - ``beta`` = pages(L) - pages(R) (left-only)
          - ``gamma`` = pages not in either R or L
          - ``U`` = all known pages

        and then calls `createLayout` with the working set appropriate to
        `scan_method`:
          - ``'add'``: add tail pages from ``gamma``, then from ``U``.
          - ``'remove'``: remove tail pages from ``beta``.
          - ``'add_round2'``: same as ``'add'`` but using head-order pages.
          - ``'auto_blind'``: fully blind mixing (`createLayout('auto',
            'blind', None)`).
          - ``'auto_reduce-max'``: delegates to `autoReduceMaximalGap`.

        Args:
            scan_method: One of the strategy names above.

        Returns:
            bool: True if a new layout was successfully created and written.
        """
        # given a two layouts: R=right and L=left:
        # alpha = hugepages(R) \ hugepages(L)
        # beta = hugepages(L) \ hugepages(R)
        # gamma = complement{hugepages(R) U hugepages(L)}
        # U = all pages
        alpha, beta, gamma, U = self.getWorkingSetPages()

        done = False

        if scan_method == 'add':
            done = done or self.createLayout('add', 'tail', gamma)
            done = done or self.createLayout('add', 'tail', U)
        elif scan_method == 'remove':
            done = done or self.createLayout('remove', 'tail', beta)
        elif scan_method == 'add_round2':
            done = done or self.createLayout('add', 'head', gamma)
            done = done or self.createLayout('add', 'head', U)
        elif scan_method == 'auto_blind':
            done = done or self.createLayout('auto', 'blind', None)
        elif scan_method == 'auto_reduce-max':
            done = done or self.autoReduceMaximalGap()
        else:
            done = self.createLayoutUsingScanMethod()

        return done

    def createLayout(self, current_direction, current_order, main_working_set, secondary_working_set=None):
        """Generates, validates, and persists one new candidate layout for the current subgroup.

        Computes the next gap to close (`StateLog.getNextBaseLayout`/
        `getNextIncrementBase`/`getNextExpectedRealCoverage`), predicts scan
        parameters (`getScanParameters`), executes them
        (`applyScanParameters`) to obtain a candidate page set, and -- if
        successful -- records the new layout in `state_log`
        (`StateLog.addRecord`), writes its Mosalloc configuration
        (`writeLayout`), and decrements the subgroup's remaining budget
        (`SubgroupsLog.decreaseRemainingBudget`).

        Args:
            current_direction: ``'add'``, ``'remove'``, or ``'auto'``.
            current_order: ``'tail'``, ``'head'``, or ``'blind'``.
            main_working_set: Primary candidate pages for this scan.
            secondary_working_set: Optional fallback candidate pages.

        Returns:
            bool: True if a novel layout was created and written to disk;
            False if this strategy failed to find one (caller should try
            the next strategy).
        """
        # keep going with the same last scan method
        #last_layout = self.state_log.getLastLayoutName()
        #last_direction = self.state_log.getLayoutScanDirection(last_layout)
        #if last_direction != 'none' and last_direction != current_direction:
        #    return False

        print('****************************************************************************')
        print(f'trying to create a new layout - method: {current_direction} , search-order: {current_order}')

        # start looking for the next gap to close in the current interval
        right_layout = self.state_log.getRightLayoutName()
        increment_base = self.state_log.getNextIncrementBase()
        base_layout = self.state_log.getNextBaseLayout(current_direction, current_order)
        expected_real_coverage = self.state_log.getNextExpectedRealCoverage()
        assert increment_base is not None

        # initialize the scan parameters based on current state
        scan_direction, scan_order, desired_pebs_coverage, base_layout = \
            self.getScanParameters(increment_base, base_layout, expected_real_coverage, current_direction, current_order)
        scan_value = desired_pebs_coverage
        assert scan_direction is not None
        assert scan_order is not None

        print('==========================================')
        print(f'{scan_direction} - {scan_order}: desired_pebs_coverage: {desired_pebs_coverage} , base_layout: {base_layout}')
        print('==========================================')

        # apply the scan and create the next layout
        pages, pebs_coverage, base_layout, increment_base, expected_real_coverage, factor = \
            self.applyScanParameters(scan_direction, scan_order, \
                desired_pebs_coverage, expected_real_coverage, \
                    base_layout, increment_base, \
                        main_working_set, secondary_working_set)
        if factor is not None:
            scan_value = factor

        if pages is None:
            print('---------------------')
            print(f'[x] could not create layout - method: {current_direction} , search-order: {current_order}')
            print('****************************************************************************')
            return False

        assert scan_direction is not None
        assert scan_order is not None
        assert base_layout is not None
        assert pebs_coverage is not None
        assert expected_real_coverage is not None
        assert increment_base is not None
        assert pages is not None

        print('+++++++++++++++++++++')
        print(f'[v] succeeded to create layout - method: {current_direction} , search-order: {current_order}')
        print('****************************************************************************')

        # update the state log by adding next generated layout
        self.state_log.addRecord(self.layout, scan_direction, scan_order,
                                 scan_value, base_layout,
                                 pebs_coverage, expected_real_coverage,
                                 increment_base, pages)
        # write the layout configuration file
        self.writeLayout(self.layout, pages)
        # decrease current group's budget by 1
        self.subgroups_log.decreaseRemainingBudget(
            self.state_log.getLeftLayoutName())

        print('----------------------------------------------')
        print(f'{self.layout} was generated with:')
        print(f'\t#hugepages: {len(pages)}')
        print(f'\tweight: {pebs_coverage}')
        print('----------------------------------------------')

        return True

class LayoutGeneratorUtils(metaclass=Singleton):
    """Stateless helpers shared across `LayoutGenerator` instances and log classes.

    A `Singleton` so that the pool footprints (set once via
    `setPoolsFootprints`) are shared process-wide, mirroring how a single
    `createLayouts.py` invocation only ever handles one benchmark/experiment.
    All other methods here are effectively static utility functions (they
    take no ``self`` beyond convention) operating on Mosalloc layout CSV
    files and PEBS dataframes.
    """
    HUGE_PAGE_2MB_SIZE = 2097152
    BASE_PAGE_4KB_SIZE = 4096

    brk_footprint = None
    mmap_footprint = None

    def __init__(self):
        pass

    def setPoolsFootprints(brk_footprint, mmap_footprint):
        """Records the benchmark's brk/mmap pool footprints for later use by `writeLayout`/`writeLayoutAll2mb`."""
        LayoutGeneratorUtils.brk_footprint = brk_footprint
        LayoutGeneratorUtils.mmap_footprint = mmap_footprint

    def loadDataframe(results_file):
        """Loads a benchmark results CSV into a compact DataFrame of layout performance metrics.

        Args:
            results_file: Path to a `collectResults.py`-produced CSV file
                (see `analysis/performance_statistics.py`).

        Returns:
            pandas.DataFrame | None: Columns ``layout``, ``walk_cycles``,
            ``stlb_hits``, ``stlb_misses``, ``cpu-cycles``, with exact
            duplicate rows removed; or ``None`` if `results_file` does not
            exist yet.
        """
        if not os.path.isfile(results_file):
            return None
        results_ps = PerformanceStatistics(results_file)
        results_df = results_ps.getDataFrame()
        results_df['cpu-cycles'] = results_ps.getRuntime()
        results_df['walk_cycles'] = results_ps.getWalkDuration()
        results_df['stlb_hits'] = results_ps.getStlbHits()
        results_df['stlb_misses'] = results_ps.getStlbMisses()
        df = results_df[['layout', 'walk_cycles', 'stlb_hits', 'stlb_misses', 'cpu-cycles']]
        # drop duplicated rows
        important_columns = list(df.columns)
        important_columns.remove('layout')
        #df.drop_duplicates(inplace=True, subset=important_columns)
        df = df.drop_duplicates(subset=important_columns)
        return df

    def writeLayoutAll2mb(layout, output):
        """Writes the trivial all-hugepages Mosalloc layout config (the entire brk pool backed by 2MB pages).

        Args:
            layout: Layout name to record in the output CSV.
            output: Destination CSV file path.
        """
        assert LayoutGeneratorUtils.brk_footprint is not None
        assert LayoutGeneratorUtils.mmap_footprint is not None

        brk_pool_size = Utils.round_up(
            LayoutGeneratorUtils.brk_footprint,
            LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE)
        configuration = Configuration()
        configuration.setPoolsSize(
                brk_size=brk_pool_size,
                file_size=1*Utils.GB,
                mmap_size=LayoutGeneratorUtils.mmap_footprint)
        configuration.addWindow(
                type=configuration.TYPE_BRK,
                page_size=LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE,
                start_offset=0,
                end_offset=brk_pool_size)
        configuration.exportToCSV(output, layout)

    def writeLayout(layout, pages, output, sliding_index=0):
        """Writes a Mosalloc layout config backing the given brk-pool pages with 2MB hugepages.

        Args:
            layout: Layout name to record in the output CSV.
            pages: Base-page numbers (relative to the brk pool) to back with
                hugepages.
            output: Destination CSV file path.
            sliding_index: Optional 4KB-page offset applied to every window,
                used to slide the brk pool's start for alignment
                experiments.
        """
        page_size= LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE
        hugepages_start_offset = sliding_index * LayoutGeneratorUtils.BASE_PAGE_4KB_SIZE
        brk_pool_size = Utils.round_up(LayoutGeneratorUtils.brk_footprint, page_size) + hugepages_start_offset
        configuration = Configuration()
        configuration.setPoolsSize(
                brk_size=brk_pool_size,
                file_size=1*Utils.GB,
                mmap_size=LayoutGeneratorUtils.mmap_footprint)
        for p in pages:
            configuration.addWindow(
                    type=configuration.TYPE_BRK,
                    page_size=page_size,
                    start_offset=(p * page_size) + hugepages_start_offset,
                    end_offset=((p+1) * page_size) + hugepages_start_offset)
        configuration.exportToCSV(output, layout)

    def getLayoutHugepages(layout_name, exp_dir):
        """Reads back the set of 2MB-hugepage-backed page numbers from a previously written layout CSV.

        Args:
            layout_name: Name of the layout (its config is read from
                ``<exp_dir>/layouts/<layout_name>.csv``).
            exp_dir: Root experiment directory.

        Returns:
            list[int]: The page numbers backed by 2MB hugepages in this
            layout.
        """
        page_size = LayoutGeneratorUtils.HUGE_PAGE_2MB_SIZE
        layout_file = str.format('{exp_root}/layouts/{layout_name}.csv',
                exp_root=exp_dir,
                layout_name=layout_name)
        df = pd.read_csv(layout_file)
        df = df[df['type'] == 'brk']
        df = df[df['pageSize'] == page_size]
        pages = []
        offset = 0
        for index, row in df.iterrows():
            start_page = int(row['startOffset'] / page_size)
            end_page = int(row['endOffset'] / page_size)
            offset = int(row['startOffset'] % page_size)
            pages += list(range(start_page, end_page))
        start_offset = offset / LayoutGeneratorUtils.BASE_PAGE_4KB_SIZE
        return pages

    def calculateTlbCoverage(pebs_df, pages):
        """Sums the PEBS `TLB_COVERAGE` of the given pages.

        Args:
            pebs_df: PEBS DataFrame with ``PAGE_NUMBER``/``TLB_COVERAGE``
                columns.
            pages: Page numbers to sum coverage for.

        Returns:
            float: Total PEBS-predicted TLB coverage percentage (0-100) of
            `pages`.
        """
        selected_pages = pebs_df.query(
                'PAGE_NUMBER in {pages}'.format(pages=pages))
        return selected_pages['TLB_COVERAGE'].sum()

    def normalizePebsAccesses(pebs_mem_bins):
        """Loads a raw PEBS mem-bins CSV and converts per-page access counts into TLB-coverage percentages.

        Args:
            pebs_mem_bins: Path to the PEBS mem-bins CSV (produced by the
                PEBS-collection tooling), expected to contain a
                ``PAGE_TYPE`` column identifying brk-pool accesses.

        Returns:
            pandas.DataFrame: Columns ``PAGE_NUMBER``, ``NUM_ACCESSES``,
            ``TLB_COVERAGE`` (the latter as a percentage of total sampled
            accesses), sorted by `TLB_COVERAGE` descending.

        Raises:
            SystemExit: If the input file has no brk-pool page-access rows.
        """
        # read mem-bins
        pebs_df = pd.read_csv(pebs_mem_bins, delimiter=',')

        # filter and eep only brk pool accesses
        pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
        if pebs_df.empty:
            sys.exit('Input file does not contain page accesses information about the brk pool!')
        pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
        pebs_df = pebs_df.reset_index()

        # transform NUM_ACCESSES from absolute number to percentage
        total_access = pebs_df['NUM_ACCESSES'].sum()
        pebs_df['TLB_COVERAGE'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        return pebs_df
