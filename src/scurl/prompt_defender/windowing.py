"""Sliding window analysis with density-based hotspot detection.

Implements a two-phase approach:
1. Coarse pass: ~4096 char windows to identify suspicious regions
2. Fine pass: ~512 char windows to drill down into hotspots

Uses density-based clustering to identify and batch hotspot regions
for efficient embedding generation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import numpy as np

from .motifs import MotifMatcher, MotifSignal


@dataclass
class Window:
    """A text window with analysis results."""
    start: int
    end: int
    text: str
    signal: MotifSignal
    score: float  # Combined suspiciousness score

    @property
    def center(self) -> int:
        """Center position of window."""
        return (self.start + self.end) // 2

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Hotspot:
    """A region identified as potentially containing injection."""
    start: int
    end: int
    peak_score: float
    windows: List[Window] = field(default_factory=list)

    @property
    def text_slice(self) -> Tuple[int, int]:
        """Return (start, end) for slicing text."""
        return (self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class AnalysisResult:
    """Complete analysis result for a text."""
    coarse_windows: List[Window]
    hotspots: List[Hotspot]
    fine_windows: List[Window]  # Windows from hotspot drill-down
    batched_regions: List[Tuple[int, int]]  # Merged regions for embedding


class SlidingWindowAnalyzer:
    """Two-phase sliding window analyzer with density-based hotspot detection.

    Phase 1 (Coarse): Scan full text with large windows (~4096 chars)
    to identify regions with elevated motif density.

    Phase 2 (Fine): Drill down into identified hotspots with smaller
    windows (~512 chars) for precise localization.

    Hotspots are then batched (merged if close) for efficient embedding.
    """

    def __init__(
        self,
        coarse_window: int = 4096,
        coarse_step: int = 2048,
        fine_window: int = 512,
        fine_step: int = 256,
        hotspot_threshold: float = 0.3,
        batch_max_size: int = 1024,
        batch_gap_tolerance: int = 256,
        motif_threshold: int = 75,
    ):
        """Initialize analyzer.

        Args:
            coarse_window: Size of coarse pass windows in chars.
            coarse_step: Step size for coarse pass.
            fine_window: Size of fine pass windows in chars.
            fine_step: Step size for fine pass.
            hotspot_threshold: Minimum score to consider a window suspicious.
            batch_max_size: Maximum size for batched regions (for embedding).
            batch_gap_tolerance: Max gap between hotspots to merge them.
            motif_threshold: Fuzzy match threshold for motif detection.
        """
        self.coarse_window = coarse_window
        self.coarse_step = coarse_step
        self.fine_window = fine_window
        self.fine_step = fine_step
        self.hotspot_threshold = hotspot_threshold
        self.batch_max_size = batch_max_size
        self.batch_gap_tolerance = batch_gap_tolerance

        self.matcher = MotifMatcher(threshold=motif_threshold)

    def analyze(self, text: str) -> AnalysisResult:
        """Run full two-phase analysis on text.

        Args:
            text: Full text to analyze.

        Returns:
            AnalysisResult with windows, hotspots, and batched regions.
        """
        # Phase 1: Coarse pass
        coarse_windows = self._scan_windows(
            text,
            window_size=self.coarse_window,
            step=self.coarse_step,
        )

        # Identify hotspots using density-based detection
        hotspots = self._detect_hotspots(coarse_windows, text)

        # Phase 2: Fine pass on hotspots
        fine_windows = []
        for hotspot in hotspots:
            # Extract hotspot region with padding
            region_start = max(0, hotspot.start - self.fine_window // 2)
            region_end = min(len(text), hotspot.end + self.fine_window // 2)
            region_text = text[region_start:region_end]

            # Scan with fine windows
            region_windows = self._scan_windows(
                region_text,
                window_size=self.fine_window,
                step=self.fine_step,
            )

            # Adjust positions to absolute offsets
            for w in region_windows:
                w.start += region_start
                w.end += region_start

            fine_windows.extend(region_windows)
            hotspot.windows = region_windows

        # Batch nearby hotspots for efficient embedding
        batched_regions = self._batch_hotspots(hotspots)

        return AnalysisResult(
            coarse_windows=coarse_windows,
            hotspots=hotspots,
            fine_windows=fine_windows,
            batched_regions=batched_regions,
        )

    def _scan_windows(
        self,
        text: str,
        window_size: int,
        step: int,
    ) -> List[Window]:
        """Scan text with sliding windows.

        Args:
            text: Text to scan.
            window_size: Window size in chars.
            step: Step between windows.

        Returns:
            List of Window objects with motif analysis.
        """
        windows = []
        text_len = len(text)

        # For very short texts, analyze the whole thing
        if text_len <= 50:
            if text_len > 0:
                signal = self.matcher.compute_signal(text)
                score = self._compute_window_score(signal)
                windows.append(Window(
                    start=0,
                    end=text_len,
                    text=text,
                    signal=signal,
                    score=score,
                ))
            return windows

        for start in range(0, text_len, step):
            end = min(start + window_size, text_len)
            window_text = text[start:end]

            if len(window_text) < 20:  # Skip very tiny trailing windows
                continue

            # Compute motif signal
            signal = self.matcher.compute_signal(window_text)

            # Calculate combined score
            score = self._compute_window_score(signal)

            windows.append(Window(
                start=start,
                end=end,
                text=window_text,
                signal=signal,
                score=score,
            ))

        return windows

    def _compute_window_score(self, signal: MotifSignal) -> float:
        """Compute suspiciousness score from motif signal.

        Combines density and category coverage into a single 0-1 score.
        """
        # Density component (0-1, caps at 5 matches per 1000 chars)
        density_score = min(signal.density / 5, 1.0)

        # Category coverage (0-1)
        category_count = len(signal.category_scores)
        coverage_score = min(category_count / 3, 1.0)  # 3+ categories = 1.0

        # Average category score (0-1)
        if signal.category_scores:
            avg_category = sum(signal.category_scores.values()) / (100 * len(signal.category_scores))
        else:
            avg_category = 0.0

        # Weighted combination
        return (
            0.4 * density_score +
            0.3 * coverage_score +
            0.3 * avg_category
        )

    def _detect_hotspots(
        self,
        windows: List[Window],
        text: str,
    ) -> List[Hotspot]:
        """Detect hotspots using density-based clustering.

        Identifies contiguous regions where window scores exceed threshold,
        merging adjacent suspicious windows into single hotspots.
        """
        if not windows:
            return []

        # Find windows above threshold
        suspicious = [w for w in windows if w.score >= self.hotspot_threshold]

        if not suspicious:
            return []

        # Sort by position
        suspicious.sort(key=lambda w: w.start)

        # Cluster adjacent windows using simple density approach
        # (DBSCAN-inspired but simpler for 1D case)
        hotspots = []
        current_hotspot_windows = [suspicious[0]]

        for window in suspicious[1:]:
            last_window = current_hotspot_windows[-1]

            # Check if this window is close to the current cluster
            gap = window.start - last_window.end
            if gap <= self.coarse_step:  # Windows overlap or touch
                current_hotspot_windows.append(window)
            else:
                # Finalize current hotspot and start new one
                hotspots.append(self._create_hotspot(current_hotspot_windows))
                current_hotspot_windows = [window]

        # Don't forget the last cluster
        if current_hotspot_windows:
            hotspots.append(self._create_hotspot(current_hotspot_windows))

        return hotspots

    def _create_hotspot(self, windows: List[Window]) -> Hotspot:
        """Create hotspot from cluster of windows."""
        return Hotspot(
            start=windows[0].start,
            end=windows[-1].end,
            peak_score=max(w.score for w in windows),
            windows=windows,
        )

    def _batch_hotspots(
        self,
        hotspots: List[Hotspot],
    ) -> List[Tuple[int, int]]:
        """Batch nearby hotspots into regions for embedding.

        Merges hotspots that are close together (within gap_tolerance)
        up to batch_max_size to reduce number of embedding calls.
        """
        if not hotspots:
            return []

        # Sort by position
        sorted_spots = sorted(hotspots, key=lambda h: h.start)

        batched = []
        current_start = sorted_spots[0].start
        current_end = sorted_spots[0].end

        for hotspot in sorted_spots[1:]:
            gap = hotspot.start - current_end
            potential_size = hotspot.end - current_start

            # Merge if close and won't exceed max size
            if gap <= self.batch_gap_tolerance and potential_size <= self.batch_max_size:
                current_end = hotspot.end
            else:
                # Finalize current batch
                batched.append((current_start, current_end))
                current_start = hotspot.start
                current_end = hotspot.end

        # Don't forget the last batch
        batched.append((current_start, current_end))

        return batched

    def get_regions_for_embedding(
        self,
        text: str,
        min_score: float = 0.0,
    ) -> List[str]:
        """Convenience method to get text regions ready for embedding.

        Args:
            text: Full text to analyze.
            min_score: Minimum hotspot score to include.

        Returns:
            List of text strings from batched hotspot regions.
        """
        result = self.analyze(text)

        regions = []
        for start, end in result.batched_regions:
            # Filter by score if specified
            if min_score > 0:
                region_score = max(
                    (h.peak_score for h in result.hotspots
                     if h.start >= start and h.end <= end),
                    default=0
                )
                if region_score < min_score:
                    continue

            regions.append(text[start:end])

        return regions


class AdaptiveWindowAnalyzer(SlidingWindowAnalyzer):
    """Analyzer with adaptive window sizing based on content.

    Automatically adjusts window sizes based on text length and
    initial signal density for more efficient analysis.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_coarse = 1024
        self._max_coarse = 8192

    def analyze(self, text: str) -> AnalysisResult:
        """Analyze with adaptive window sizing."""
        text_len = len(text)

        # Short texts: single pass with smaller windows
        if text_len <= self._min_coarse:
            self.coarse_window = text_len
            self.coarse_step = text_len
            self.fine_window = min(512, text_len)
            self.fine_step = min(256, text_len // 2)

        # Medium texts: use configured defaults
        elif text_len <= self._max_coarse * 2:
            pass  # Use initialized values

        # Long texts: increase coarse window for efficiency
        else:
            self.coarse_window = min(self._max_coarse, text_len // 4)
            self.coarse_step = self.coarse_window // 2

        return super().analyze(text)
