# -*- coding: utf-8 -*-
"""
Spike Data Quality Check GUI v11
Changes from v10:
  - NEV loading is now lazy and chunked:
      • Opening a .nev file reads only the channel headers (fast).  The GUI
        becomes interactive immediately, showing all channel rows with a
        "Loading…" placeholder.
      • Spike times and waveforms are loaded 32 channels at a time in a
        background QThread as you page through the data.
      • Already-loaded channels are cached, so revisiting a page is instant.
      • A progress bar + status label in the top bar shows which chunk is
        being loaded.  Navigation is disabled while a chunk loads, then
        re-enabled automatically.
      • PKL files are still loaded all at once (they are already
        serialised numpy arrays and load in seconds).

Author: claude
"""

import sys
import os
import pickle
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QFileDialog,
    QScrollBar, QMessageBox, QScrollArea, QCheckBox, QSizePolicy, QFrame,
    QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── BMI / pyns path setup ─────────────────────────────────────────────────────
BMI_FOLDER = r"C:\Users\coleb\Desktop\Santacruz Lab\bmi_python"
NS_FOLDER  = os.path.join(BMI_FOLDER, 'riglib', 'ripple', 'pyns', 'pyns')
sys.path.insert(1, BMI_FOLDER)
sys.path.insert(2, NS_FOLDER)

# ── Constants ──────────────────────────────────────────────────────────────────
SORTED_COLOR   = "#d0e8ff"
UNSORTED_COLOR = "#e8e8e8"
FR_WINDOW_SEC  = 60.0

WF_EXCL_COLOR  = "#cccccc"
WF_INCL_COLOR  = "#6495ED"
WF_MEAN_COLOR  = "#00008B"

UB_COLOR = "#cc0000"     # upper bound line colour (solid red)
LB_COLOR = "#e07000"     # lower bound line colour (solid orange)

ROW_HEIGHT = 300
PLOT_H     = ROW_HEIGHT - 16


# ──────────────────────────────────────────────────────────────────────────────
class ChunkLoader(QThread):
    """Background thread that loads spike data for a contiguous slice of units.

    Emits `chunk_ready` with (start_idx, list_of_(times_array, waves_array))
    when the slice is done, or `error` with a traceback string on failure.
    """
    chunk_ready = pyqtSignal(int, list)   # (page_start, [(times, waves), …])
    progress    = pyqtSignal(int, int)    # (done_count, total_count)
    error       = pyqtSignal(str)

    def __init__(self, spike_entities, valid_idxs, start, count):
        super().__init__()
        self._spike_entities = spike_entities
        self._valid_idxs     = valid_idxs   # full list; we slice [start:start+count]
        self._start          = start        # index into valid_idxs
        self._count          = count

    def run(self):
        try:
            slice_idxs = self._valid_idxs[self._start : self._start + self._count]
            results    = []
            for done, entity_idx in enumerate(slice_idxs):
                unit   = self._spike_entities[entity_idx]
                n      = unit.item_count

                if n == 0:
                    results.append((np.array([]), np.empty((0, 0))))
                    self.progress.emit(done + 1, len(slice_idxs))
                    continue

                # Pre-fetch first spike to learn waveform length
                seg0   = unit.get_segment_data(0)
                n_samp = len(seg0[1])

                times  = np.empty(n,              dtype=np.float64)
                waves  = np.empty((n, n_samp),    dtype=np.float64)
                times[0]  = seg0[0]
                waves[0]  = seg0[1]

                for k in range(1, n):
                    seg       = unit.get_segment_data(k)
                    times[k]  = seg[0]
                    waves[k]  = seg[1]

                results.append((times, waves))
                self.progress.emit(done + 1, len(slice_idxs))

            self.chunk_ready.emit(self._start, results)
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ──────────────────────────────────────────────────────────────────────────────
class SpikeQualityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spike Quality Check GUI v11")
        self.setWindowIcon(QIcon(
            r"C:\Users\coleb\Desktop\Santacruz Lab\Resources"
            r"\SantacruzLab_FINAL_PPT-01-icon only.png"))
        self.setGeometry(100, 50, 1800, 950)

        # ── raw data ──────────────────────────────────────────────────────────
        self.unit_labels        = []
        self.sorted_mask        = []
        self.raw_spike_times    = {}   # label -> np.array (s)
        self.raw_waveforms      = {}   # label -> np.array (n_spikes × n_samples)
        self.recording_duration = 0

        # ── NEV lazy-load state ───────────────────────────────────────────────
        # Set when a .nev is opened; cleared when a .pkl is loaded instead.
        self._nev_spike_entities = None  # list of spike entity objects
        self._nev_valid_idxs     = None  # list of entity indices (parallel to unit_labels)
        self._nev_loaded_set     = set() # set of label indices already in raw_* dicts
        self._chunk_loader       = None  # active QThread or None

        # ── per-unit persistent state ─────────────────────────────────────────
        self.excl_param    = {}   # label -> float  (N std devs, fallback)
        self.good_state    = {}   # label -> bool
        self.upper_bound   = {}   # label -> np.array (n_samples,) or None
        self.lower_bound   = {}   # label -> np.array (n_samples,) or None
        self.wf_ylim       = {}   # label -> None (unused legacy; kept for zoom)

        # ── active draw session ───────────────────────────────────────────────
        # which = 'upper' | 'lower' | None
        self._draw_label   = None
        self._draw_which   = None
        self._draw_pts_x   = []
        self._draw_pts_y   = []
        self._draw_line    = None   # live preview Line2D

        # ── display params ────────────────────────────────────────────────────
        self.isi_bin_ms   = 5.0
        self.isi_max_ms   = 250.0
        self.waveform_nth = 50

        self.row_widgets        = []
        self.units_per_page     = 32
        self.current_page_start = 0

        # ── source file info (for export dialog defaults) ─────────────────────
        self.loaded_dir     = ""   # folder the data file came from
        self.loaded_session = ""   # bare filename stem (no extension)

        self._build_ui()

    # =========================================================================
    # UI construction
    # =========================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)

        ctrl = QHBoxLayout()

        btn_nev = QPushButton("Browse .nev file…")
        btn_nev.clicked.connect(self.load_nev_file)
        ctrl.addWidget(btn_nev)

        btn_pkl = QPushButton("Load spike dict .pkl…")
        btn_pkl.clicked.connect(self.load_pkl_file)
        ctrl.addWidget(btn_pkl)

        ctrl.addWidget(QLabel("  ISI Bin (ms):"))
        self.isi_bin_spin = QDoubleSpinBox()
        self.isi_bin_spin.setRange(0.1, 50.0)
        self.isi_bin_spin.setValue(5.0)
        self.isi_bin_spin.setSingleStep(0.5)
        self.isi_bin_spin.setDecimals(1)
        self.isi_bin_spin.valueChanged.connect(self._on_isi_param_changed)
        ctrl.addWidget(self.isi_bin_spin)

        ctrl.addWidget(QLabel("ISI Max (ms):"))
        self.isi_max_spin = QDoubleSpinBox()
        self.isi_max_spin.setRange(10.0, 5000.0)
        self.isi_max_spin.setValue(250.0)
        self.isi_max_spin.setSingleStep(10.0)
        self.isi_max_spin.setDecimals(1)
        self.isi_max_spin.valueChanged.connect(self._on_isi_param_changed)
        ctrl.addWidget(self.isi_max_spin)

        ctrl.addWidget(QLabel("Waveform N:"))
        self.wf_nth_combo = QComboBox()
        for opt in ["50", "100", "500", "1000"]:
            self.wf_nth_combo.addItem(opt)
        self.wf_nth_combo.setCurrentText("50")
        self.wf_nth_combo.setToolTip("Total number of waveforms to plot per unit")
        self.wf_nth_combo.currentTextChanged.connect(self._on_wf_nth_changed)
        ctrl.addWidget(self.wf_nth_combo)

        ctrl.addStretch()

        self.load_status_lbl = QLabel("")
        self.load_status_lbl.setStyleSheet("color:#555; font-style:italic;")
        ctrl.addWidget(self.load_status_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedWidth(140)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        ctrl.addWidget(self.progress_bar)

        self.unit_range_lbl = QLabel("No data loaded")
        ctrl.addWidget(self.unit_range_lbl)

        btn_export = QPushButton("Export to .pkl")
        btn_export.setStyleSheet(
            "font-weight:bold; background:#4a9; color:white; padding:4px 12px;")
        btn_export.clicked.connect(self.export_good_units)
        ctrl.addWidget(btn_export)
        root.addLayout(ctrl)

        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Previous page")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._on_prev_page)
        nav_row.addWidget(self.btn_prev)

        self.page_lbl = QLabel("–")
        self.page_lbl.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.page_lbl)

        self.btn_next = QPushButton("Next page  ▶")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next_page)
        nav_row.addWidget(self.btn_next)
        root.addLayout(nav_row)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.plots_widget = QWidget()
        self.plots_layout = QVBoxLayout(self.plots_widget)
        self.plots_layout.setSpacing(6)
        self.plots_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_area.setWidget(self.plots_widget)
        root.addWidget(self.scroll_area)

    # =========================================================================
    # Helpers
    # =========================================================================
    def _visible_labels(self):
        return list(self.unit_labels)

    # =========================================================================
    # Data loading
    # =========================================================================
    def load_nev_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select .nev file", "", "NEV Files (*.nev)")
        if not path:
            return
        try:
            from nsfile import NSFile
            print(f"Reading headers from {path} …")
            nev     = NSFile(path)
            session = os.path.basename(path)[:-4]
            self.loaded_dir     = os.path.dirname(path)
            self.loaded_session = session

            # ── fast header scan only ──────────────────────────────────────
            spike_entities   = [e for e in nev.get_entities() if e.entity_type == 3]
            headers          = np.array([s.get_extended_headers() for s in spike_entities])
            all_idxs         = np.arange(len(headers))
            sorted_unit_idxs = set(np.nonzero(
                [h[b'NEUEVWAV'].number_sorted_units for h in headers])[0])
            valid_idxs = [i for i in all_idxs if b'NEUEVLBL' in headers[i].keys()]

            unit_labels = []
            sorted_mask = []
            for idx in valid_idxs:
                h   = headers[idx]
                lbl = "Unit " + h[b'NEUEVLBL'].label[:7].decode() + session
                unit_labels.append(lbl)
                sorted_mask.append(idx in sorted_unit_idxs)

            rec_dur = nev.get_file_info().time_span

            # ── store NEV state for lazy loading ──────────────────────────
            self._nev_spike_entities = spike_entities
            self._nev_valid_idxs     = valid_idxs
            self._nev_loaded_set     = set()

            # Initialise metadata without spike data (empty placeholders)
            self._init_metadata(unit_labels, sorted_mask, rec_dur)

            print(f"Headers loaded: {len(unit_labels)} channels. "
                  f"Loading first page of spike data…")

            # Kick off loading the first page immediately
            self._load_chunk(0)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"{e}\n\n{traceback.format_exc()}")

    def load_pkl_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spike dict .pkl", "", "PKL Files (*.pkl)")
        if not path:
            return
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            self.loaded_dir     = os.path.dirname(path)
            self.loaded_session = os.path.basename(path)[:-4]
            labels      = d['unit_labels']
            times       = d['spike_times']
            waves       = d['spike_waveforms']
            sorted_mask = d.get('sorted_mask', [True] * len(labels))
            rec_dur     = d.get('recording_duration', 0)
            if rec_dur == 0:
                all_t   = np.concatenate([np.array(t) for t in times])
                rec_dur = float(np.max(all_t))
            self._store_data(labels, sorted_mask, times, waves, rec_dur)
            QMessageBox.information(self, "Loaded",
                f"PKL loaded. {len(labels)} units. Duration: {rec_dur:.1f} s")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"{e}\n\n{traceback.format_exc()}")

    def _init_metadata(self, labels, sorted_mask, rec_dur):
        """Set up all per-unit dicts with empty placeholders (no spike data yet).
        Called immediately after reading NEV headers so the GUI can render rows.
        """
        self.unit_labels        = labels
        self.sorted_mask        = sorted_mask
        self.recording_duration = rec_dur

        self.raw_spike_times = {lbl: np.array([]) for lbl in labels}
        self.raw_waveforms   = {lbl: np.empty((0, 0)) for lbl in labels}
        self.excl_param      = {lbl: 3.0  for lbl in labels}
        self.good_state      = {lbl: bool(s) for lbl, s in zip(labels, sorted_mask)}
        self.upper_bound     = {lbl: None for lbl in labels}
        self.lower_bound     = {lbl: None for lbl in labels}
        self.wf_ylim         = {lbl: None for lbl in labels}

        self.current_page_start = 0
        self._update_nav_buttons()
        self._build_rows()
        self._update_all_plots()   # will show "Not enough spikes" placeholders
        self._update_range_label()

    def _load_chunk(self, page_start):
        """Start a background ChunkLoader for the 32 channels at page_start."""
        if self._nev_spike_entities is None:
            return   # PKL mode — nothing to do
        if self._chunk_loader is not None and self._chunk_loader.isRunning():
            return   # already loading

        n        = len(self.unit_labels)
        count    = min(self.units_per_page, n - page_start)
        if count <= 0:
            return

        # Check which channels in this slice still need loading
        already_done = all(i in self._nev_loaded_set
                           for i in range(page_start, page_start + count))
        if already_done:
            return   # cache hit — nothing to fetch

        self.load_status_lbl.setText(
            f"Loading channels {page_start+1}–{page_start+count} of {n} …")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

        loader = ChunkLoader(
            self._nev_spike_entities,
            self._nev_valid_idxs,
            page_start,
            count)
        loader.progress.connect(self._on_chunk_progress)
        loader.chunk_ready.connect(self._on_chunk_ready)
        loader.error.connect(self._on_chunk_error)
        self._chunk_loader = loader
        loader.start()

    def _on_chunk_progress(self, done, total):
        pct = int(done / total * 100)
        self.progress_bar.setValue(pct)
        self.load_status_lbl.setText(
            f"Loading … {done}/{total} channels")

    def _on_chunk_ready(self, page_start, results):
        """Called in the GUI thread when a chunk finishes loading."""
        for offset, (times, waves) in enumerate(results):
            global_idx = page_start + offset
            if global_idx >= len(self.unit_labels):
                break
            lbl = self.unit_labels[global_idx]
            self.raw_spike_times[lbl] = times
            self.raw_waveforms[lbl]   = waves
            self._nev_loaded_set.add(global_idx)

        self.progress_bar.setVisible(False)
        self.load_status_lbl.setText(
            f"Channels {page_start+1}–{page_start+len(results)} loaded ✓")
        self._chunk_loader = None
        self._update_nav_buttons()

        # Redraw current page rows with real data
        self._update_all_plots()

    def _on_chunk_error(self, tb):
        self.progress_bar.setVisible(False)
        self.load_status_lbl.setText("Load error — see console")
        self._chunk_loader = None
        self._update_nav_buttons()
        QMessageBox.critical(self, "Load error", tb)

    def _store_data(self, labels, sorted_mask, spike_times, spike_waveforms, rec_dur):
        """Used by PKL loading — all data available upfront."""
        # Clear any NEV lazy-load state
        self._nev_spike_entities = None
        self._nev_valid_idxs     = None
        self._nev_loaded_set     = set()
        self._chunk_loader       = None

        self.unit_labels        = labels
        self.sorted_mask        = sorted_mask
        self.recording_duration = rec_dur

        self.raw_spike_times = {}
        self.raw_waveforms   = {}
        self.excl_param      = {}
        self.good_state      = {}
        self.upper_bound     = {}
        self.lower_bound     = {}
        self.wf_ylim         = {}

        for lbl, times, waves, is_sorted in zip(
                labels, spike_times, spike_waveforms, sorted_mask):
            self.raw_spike_times[lbl] = np.array(times)
            self.raw_waveforms[lbl]   = np.array(waves)
            self.excl_param[lbl]      = 3.0
            self.good_state[lbl]      = bool(is_sorted)
            self.upper_bound[lbl]     = None
            self.lower_bound[lbl]     = None
            self.wf_ylim[lbl]         = None

        self.current_page_start = 0
        self._update_nav_buttons()
        self._build_rows()
        self._update_all_plots()
        self._update_range_label()

    # =========================================================================
    # Exclusion logic
    # =========================================================================
    def _get_keep_mask(self, label):
        """True = spike passes exclusion.

        Priority:
          1. If at least one drawn bound exists → use drawn bounds (other side ±inf)
          2. Otherwise → mean ± N·σ
        """
        waves = self.raw_waveforms[label]
        if len(waves) == 0 or waves.ndim < 2:
            return np.ones(len(waves), dtype=bool)

        ub = self.upper_bound[label]
        lb = self.lower_bound[label]

        if ub is not None or lb is not None:
            mask = np.ones(len(waves), dtype=bool)
            if ub is not None:
                mask &= np.all(waves <= ub, axis=1)
            if lb is not None:
                mask &= np.all(waves >= lb, axis=1)
            return mask
        else:
            n_std  = self.excl_param[label]
            mean_w = np.mean(waves, axis=0)
            std_w  = np.std(waves, axis=0)
            lo     = mean_w - n_std * std_w
            hi     = mean_w + n_std * std_w
            return np.all((waves >= lo) & (waves <= hi), axis=1)

    def _get_filtered(self, label):
        mask = self._get_keep_mask(label)
        return self.raw_spike_times[label][mask], self.raw_waveforms[label][mask]

    # =========================================================================
    # Row building
    # =========================================================================
    def _clear_rows(self):
        for rw in self.row_widgets:
            rw['container'].setParent(None)
        self.row_widgets = []

    def _build_rows(self):
        self._clear_rows()
        visible = self._visible_labels()
        n_show  = min(self.units_per_page, len(visible) - self.current_page_start)

        for slot in range(n_show):
            label      = visible[self.current_page_start + slot]
            global_idx = self.unit_labels.index(label)
            is_sorted  = self.sorted_mask[global_idx]
            bg_color   = SORTED_COLOR if is_sorted else UNSORTED_COLOR

            container = QFrame()
            container.setFrameShape(QFrame.StyledPanel)
            container.setStyleSheet(f"background-color:{bg_color}; border-radius:4px;")
            container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            container.setFixedHeight(ROW_HEIGHT)

            row_layout = QHBoxLayout(container)
            row_layout.setContentsMargins(4, 4, 4, 4)
            row_layout.setSpacing(6)

            # ── sidebar ───────────────────────────────────────────────────────
            left = QVBoxLayout()
            left.setSpacing(4)

            chk_good = QCheckBox("Good unit")
            chk_good.setChecked(self.good_state[label])
            chk_good.setStyleSheet("font-weight:bold; color:#006600;")
            chk_good.stateChanged.connect(
                lambda state, lbl=label: self._on_good_changed(lbl, state))
            left.addWidget(chk_good)

            lbl_w = QLabel(label)
            lbl_w.setWordWrap(True)
            lbl_w.setFixedWidth(148)
            left.addWidget(lbl_w)

            left.addWidget(QLabel("Excl ±N·σ (fallback):"))
            excl_spin = QDoubleSpinBox()
            excl_spin.setRange(0.5, 20.0)
            excl_spin.setSingleStep(0.5)
            excl_spin.setDecimals(1)
            excl_spin.setValue(self.excl_param[label])
            excl_spin.setFixedWidth(80)
            excl_spin.setToolTip(
                "Used only when no drawn bounds exist.\n"
                "Excludes spikes outside mean ± N·σ.")
            excl_spin.valueChanged.connect(
                lambda val, lbl=label, s=slot: self._on_excl_changed(lbl, val, s))
            left.addWidget(excl_spin)

            # draw-mode status
            draw_status = QLabel("")
            draw_status.setAlignment(Qt.AlignCenter)
            draw_status.setStyleSheet("font-weight:bold;")
            left.addWidget(draw_status)

            # Upper bound button
            btn_upper = QPushButton("▲ Draw upper bound")
            btn_upper.setCheckable(True)
            btn_upper.setStyleSheet(
                f"QPushButton:checked {{background:{UB_COLOR}; color:white; font-weight:bold;}}")
            btn_upper.setToolTip(
                "Click to enter draw mode, then click-drag on the waveform\n"
                "to paint the upper exclusion boundary.\n"
                "Waveforms that exceed this line at any sample are excluded.")
            btn_upper.toggled.connect(
                lambda chk, lbl=label, s=slot:
                    self._on_bound_toggled(lbl, s, 'upper', chk))
            left.addWidget(btn_upper)

            # Lower bound button
            btn_lower = QPushButton("▼ Draw lower bound")
            btn_lower.setCheckable(True)
            btn_lower.setStyleSheet(
                f"QPushButton:checked {{background:{LB_COLOR}; color:white; font-weight:bold;}}")
            btn_lower.setToolTip(
                "Click to enter draw mode, then click-drag on the waveform\n"
                "to paint the lower exclusion boundary.\n"
                "Waveforms that go below this line at any sample are excluded.")
            btn_lower.toggled.connect(
                lambda chk, lbl=label, s=slot:
                    self._on_bound_toggled(lbl, s, 'lower', chk))
            left.addWidget(btn_lower)

            # Clear bounds button
            btn_clear = QPushButton("✕ Clear bounds")
            btn_clear.setToolTip("Remove all drawn bounds and revert to ±N·σ exclusion.")
            btn_clear.clicked.connect(
                lambda _, lbl=label, s=slot: self._on_clear_bounds(lbl, s))
            left.addWidget(btn_clear)

            left.addStretch()
            row_layout.addLayout(left)

            # ── waveform canvas + y-zoom scrollbar ────────────────────────────
            wf_fig    = Figure(tight_layout=True)
            wf_canvas = FigureCanvas(wf_fig)
            wf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            wf_canvas.setFixedHeight(PLOT_H)
            wf_ax = wf_fig.add_subplot(111)

            wf_zoom_sb = QScrollBar(Qt.Vertical)
            wf_zoom_sb.setRange(1, 100)
            wf_zoom_sb.setValue(100)
            wf_zoom_sb.setFixedWidth(16)
            wf_zoom_sb.setFixedHeight(PLOT_H)
            wf_zoom_sb.setToolTip("Scroll to zoom y-axis (top=in, bottom=auto)")
            wf_zoom_sb.valueChanged.connect(
                lambda val, lbl=label: self._on_wf_zoom_sb(val, lbl))

            wf_panel = QWidget()
            wf_pl    = QHBoxLayout(wf_panel)
            wf_pl.setContentsMargins(0, 0, 0, 0)
            wf_pl.setSpacing(2)
            wf_pl.addWidget(wf_canvas)
            wf_pl.addWidget(wf_zoom_sb)
            row_layout.addWidget(wf_panel, stretch=4)

            # mpl mouse events for boundary drawing
            wf_fig.canvas.mpl_connect(
                'button_press_event',
                lambda evt, lbl=label: self._on_wf_press(evt, lbl))
            wf_fig.canvas.mpl_connect(
                'motion_notify_event',
                lambda evt, lbl=label: self._on_wf_motion(evt, lbl))
            wf_fig.canvas.mpl_connect(
                'button_release_event',
                lambda evt, lbl=label: self._on_wf_release(evt, lbl))

            # ── ISI canvas ────────────────────────────────────────────────────
            isi_fig    = Figure(tight_layout=True)
            isi_canvas = FigureCanvas(isi_fig)
            isi_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            isi_canvas.setFixedHeight(PLOT_H)
            isi_ax = isi_fig.add_subplot(111)
            row_layout.addWidget(isi_canvas, stretch=2)

            # ── FR canvas ─────────────────────────────────────────────────────
            fr_fig    = Figure(tight_layout=True)
            fr_canvas = FigureCanvas(fr_fig)
            fr_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            fr_canvas.setFixedHeight(PLOT_H)
            fr_ax = fr_fig.add_subplot(111)
            row_layout.addWidget(fr_canvas, stretch=2)

            self.plots_layout.addWidget(container)

            # Restore draw-status label if bounds already exist
            has_ub = self.upper_bound[label] is not None
            has_lb = self.lower_bound[label] is not None
            if has_ub and has_lb:
                draw_status.setText("UB + LB set ✓")
                draw_status.setStyleSheet("font-weight:bold; color:#006600;")
            elif has_ub:
                draw_status.setText("Upper bound set ✓")
                draw_status.setStyleSheet(f"font-weight:bold; color:{UB_COLOR};")
            elif has_lb:
                draw_status.setText("Lower bound set ✓")
                draw_status.setStyleSheet(f"font-weight:bold; color:{LB_COLOR};")

            self.row_widgets.append(dict(
                container=container,
                label=label,
                chk_good=chk_good,
                excl_spin=excl_spin,
                btn_upper=btn_upper,
                btn_lower=btn_lower,
                btn_clear=btn_clear,
                draw_status=draw_status,
                wf_fig=wf_fig,   wf_canvas=wf_canvas,   wf_ax=wf_ax,
                wf_zoom_sb=wf_zoom_sb,
                isi_fig=isi_fig, isi_canvas=isi_canvas, isi_ax=isi_ax,
                fr_fig=fr_fig,   fr_canvas=fr_canvas,   fr_ax=fr_ax,
            ))

    # =========================================================================
    # Plotting
    # =========================================================================
    def _update_all_plots(self):
        for slot in range(len(self.row_widgets)):
            self._plot_row(slot)

    def _plot_row(self, slot):
        rw    = self.row_widgets[slot]
        label = rw['label']
        times_f, _ = self._get_filtered(label)
        self._plot_waveform(rw)
        self._plot_isi(rw, times_f)
        self._plot_fr(rw, times_f)

    def _plot_waveform(self, rw):
        ax    = rw['wf_ax']
        label = rw['label']
        ax.clear()

        all_waves = self.raw_waveforms[label]
        if len(all_waves) == 0 or all_waves.ndim < 2:
            ax.text(0.5, 0.5, 'No waveforms',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            rw['wf_canvas'].draw()
            return

        keep_mask   = self._get_keep_mask(label)
        xs          = np.arange(all_waves.shape[1])
        N_total     = max(1, self.waveform_nth)
        excl_all    = np.where(~keep_mask)[0]
        incl_all    = np.where(keep_mask)[0]
        n_excl_all  = len(excl_all)
        n_incl_all  = len(incl_all)
        n_total_raw = n_excl_all + n_incl_all

        if n_total_raw == 0:
            rw['wf_canvas'].draw()
            return

        n_incl_show = (max(1, round(N_total * n_incl_all / n_total_raw))
                       if n_incl_all else 0)
        n_excl_show = (max(1, round(N_total * n_excl_all / n_total_raw))
                       if n_excl_all else 0)

        def subsample(idxs, n):
            if len(idxs) <= n:
                return idxs
            step = len(idxs) / n
            return idxs[np.round(np.arange(n) * step).astype(int)]

        for i in subsample(excl_all, n_excl_show):
            ax.plot(xs, all_waves[i], color=WF_EXCL_COLOR,
                    lw=0.5, alpha=0.6, zorder=1)
        for i in subsample(incl_all, n_incl_show):
            ax.plot(xs, all_waves[i], color=WF_INCL_COLOR,
                    lw=0.5, alpha=0.5, zorder=2)

        incl_waves = all_waves[keep_mask]
        n_kept = len(incl_waves)
        n_excl = len(all_waves) - n_kept

        if n_kept > 0:
            mean_w = np.mean(incl_waves, axis=0)
            ax.plot(xs, mean_w, color=WF_MEAN_COLOR, lw=2.5, zorder=3)

        # ── exclusion boundaries ──────────────────────────────────────────────
        ub = self.upper_bound[label]
        lb = self.lower_bound[label]

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color=WF_INCL_COLOR, lw=1.5, label=f'Included (N={n_kept})'),
            Line2D([0], [0], color=WF_EXCL_COLOR, lw=1.5, label=f'Excluded (N={n_excl})'),
            Line2D([0], [0], color=WF_MEAN_COLOR, lw=2.5, label='Mean'),
        ]

        if ub is not None or lb is not None:
            # Drawn boundaries take priority — show as solid coloured lines
            if ub is not None:
                ax.plot(xs, ub, color=UB_COLOR, lw=1.5, zorder=5)
                handles.append(Line2D([0], [0], color=UB_COLOR, lw=1.5,
                                      label='Upper bound'))
            if lb is not None:
                ax.plot(xs, lb, color=LB_COLOR, lw=1.5, zorder=5)
                handles.append(Line2D([0], [0], color=LB_COLOR, lw=1.5,
                                      label='Lower bound'))
        else:
            # Fallback: show ±N·σ as dashed red
            if n_kept > 0:
                std_w = np.std(incl_waves, axis=0)
                n_std = self.excl_param[label]
                ax.plot(xs, mean_w + n_std * std_w, color='red',
                        lw=0.8, linestyle='--', zorder=4, alpha=0.8)
                ax.plot(xs, mean_w - n_std * std_w, color='red',
                        lw=0.8, linestyle='--', zorder=4, alpha=0.8)
                handles.append(Line2D([0], [0], color='red', lw=0.8,
                                      linestyle='--',
                                      label=f'±{self.excl_param[label]:.1f}σ'))

        ax.legend(handles=handles, fontsize=7, loc='upper right')
        ax.set_xlabel('Sample', fontsize=8)
        ax.set_ylabel('Amplitude (µV)', fontsize=8)
        ax.set_title(f'{N_total} waveforms shown  kept={n_kept}  excl={n_excl}',
                     fontsize=8, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

        rw['wf_canvas'].draw()
        self._apply_wf_zoom(rw)
        rw['wf_canvas'].draw()

    def _plot_isi(self, rw, times):
        ax = rw['isi_ax']
        ax.clear()
        if len(times) < 2:
            ax.text(0.5, 0.5, 'Not enough spikes',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            rw['isi_canvas'].draw()
            return

        isis_ms    = np.diff(times) * 1000.0
        bins       = np.arange(0, self.isi_max_ms + self.isi_bin_ms, self.isi_bin_ms)
        ax.hist(isis_ms, bins=bins, color='steelblue',
                edgecolor='white', alpha=0.85, linewidth=0.5)
        mean_isi   = np.mean(isis_ms)
        median_isi = np.median(isis_ms)
        ax.axvline(mean_isi,   color='darkorange',    lw=2,
                   label=f'Mean ISI = {mean_isi:.1f} ms')
        ax.axvline(median_isi, color='mediumseagreen', lw=2, linestyle='--',
                   label=f'Median ISI = {median_isi:.1f} ms')

        ax.set_xlim([0, self.isi_max_ms])
        ax.set_xlabel('ISI (ms)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title('ISI Histogram', fontsize=8, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        rw['isi_canvas'].draw()

    def _plot_fr(self, rw, times):
        ax = rw['fr_ax']
        ax.clear()
        if len(times) < 2:
            ax.text(0.5, 0.5, 'Not enough spikes',
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            rw['fr_canvas'].draw()
            return

        rec_dur = (self.recording_duration if self.recording_duration > 0
                   else float(np.max(times)))
        win     = FR_WINDOW_SEC
        step    = win / 2.0
        t_cen   = np.arange(win / 2, rec_dur - win / 2 + step, step)
        fr_vals = np.array([
            np.sum((times >= tc - win / 2) & (times < tc + win / 2)) / win
            for tc in t_cen])
        mean_fr = len(times) / rec_dur

        ax.plot(t_cen / 60.0, fr_vals, color='purple', lw=1.5)
        ax.axhline(mean_fr, color='gray', lw=1, linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.set_ylabel('FR (Hz)', fontsize=8)
        ax.set_title(f'Sliding FR  (mean = {mean_fr:.2f} Hz)',
                     fontsize=8, fontweight='bold')
        ax.set_xlim([0, rec_dur / 60.0])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        rw['fr_canvas'].draw()

    # =========================================================================
    # Y-axis zoom
    # =========================================================================
    def _apply_wf_zoom(self, rw):
        ax = rw['wf_ax']
        sb = rw['wf_zoom_sb']
        ymin_auto, ymax_auto = ax.get_ylim()
        half_auto = max(abs(ymin_auto), abs(ymax_auto))
        fraction  = sb.value() / 100.0
        half_zoom = half_auto * fraction
        ax.set_ylim(-half_zoom, half_zoom)

    def _on_wf_zoom_sb(self, val, label):
        rw = next((r for r in self.row_widgets if r['label'] == label), None)
        if rw:
            self._plot_waveform(rw)

    # =========================================================================
    # Boundary draw-mode
    # =========================================================================
    def _cancel_active_draw(self):
        """Cancel any in-progress draw session without saving."""
        if self._draw_label is None:
            return
        rw = next((r for r in self.row_widgets
                   if r['label'] == self._draw_label), None)
        if rw:
            for btn_key in ('btn_upper', 'btn_lower'):
                rw[btn_key].blockSignals(True)
                rw[btn_key].setChecked(False)
                rw[btn_key].blockSignals(False)
            rw['wf_canvas'].setCursor(Qt.ArrowCursor)
            self._update_draw_status(rw)
        self._draw_label = None
        self._draw_which = None
        self._draw_pts_x = []
        self._draw_pts_y = []
        self._draw_line  = None

    def _update_draw_status(self, rw):
        """Refresh the status label based on current bound state (no active draw)."""
        label  = rw['label']
        has_ub = self.upper_bound[label] is not None
        has_lb = self.lower_bound[label] is not None
        sl     = rw['draw_status']
        if has_ub and has_lb:
            sl.setText("UB + LB set ✓")
            sl.setStyleSheet("font-weight:bold; color:#006600;")
        elif has_ub:
            sl.setText("Upper bound set ✓")
            sl.setStyleSheet(f"font-weight:bold; color:{UB_COLOR};")
        elif has_lb:
            sl.setText("Lower bound set ✓")
            sl.setStyleSheet(f"font-weight:bold; color:{LB_COLOR};")
        else:
            sl.setText("")
            sl.setStyleSheet("font-weight:bold;")

    def _on_bound_toggled(self, label, slot, which, checked):
        rw = next((r for r in self.row_widgets if r['label'] == label), None)
        if rw is None:
            return

        if checked:
            # Cancel any other active draw first
            self._cancel_active_draw()

            self._draw_label = label
            self._draw_which = which
            self._draw_pts_x = []
            self._draw_pts_y = []
            self._draw_line  = None

            color = UB_COLOR if which == 'upper' else LB_COLOR
            rw['draw_status'].setText(
                f"{'UPPER' if which == 'upper' else 'LOWER'} DRAW MODE")
            rw['draw_status'].setStyleSheet(
                f"font-weight:bold; color:{color};")
            rw['wf_canvas'].setCursor(Qt.CrossCursor)
        else:
            # De-toggled without completing a draw — just cancel
            if self._draw_label == label and self._draw_which == which:
                self._cancel_active_draw()

    def _on_wf_press(self, event, label):
        if self._draw_label != label:
            return
        if event.inaxes is None or event.button != 1:
            return
        self._draw_pts_x = [event.xdata]
        self._draw_pts_y = [event.ydata]

    def _on_wf_motion(self, event, label):
        if self._draw_label != label:
            return
        if event.inaxes is None or event.button != 1:
            return
        if not self._draw_pts_x:
            return
        self._draw_pts_x.append(event.xdata)
        self._draw_pts_y.append(event.ydata)

        rw = next((r for r in self.row_widgets if r['label'] == label), None)
        if rw is None:
            return
        color = UB_COLOR if self._draw_which == 'upper' else LB_COLOR
        ax    = rw['wf_ax']
        if self._draw_line is None:
            self._draw_line, = ax.plot(self._draw_pts_x, self._draw_pts_y,
                                       color=color, lw=1.5, zorder=10)
        else:
            self._draw_line.set_xdata(self._draw_pts_x)
            self._draw_line.set_ydata(self._draw_pts_y)
        rw['wf_canvas'].draw_idle()

    def _on_wf_release(self, event, label):
        if self._draw_label != label:
            return
        if event.button != 1 or len(self._draw_pts_x) < 2:
            return

        rw = next((r for r in self.row_widgets if r['label'] == label), None)
        if rw is None:
            return

        n_samples = self.raw_waveforms[label].shape[1]
        xs_data   = np.clip(np.array(self._draw_pts_x), 0, n_samples - 1)
        ys_data   = np.array(self._draw_pts_y)

        order     = np.argsort(xs_data)
        xs_data   = xs_data[order]
        ys_data   = ys_data[order]

        _, uid    = np.unique(xs_data, return_index=True)
        xs_data   = xs_data[uid]
        ys_data   = ys_data[uid]

        sample_xs = np.arange(n_samples, dtype=float)
        bound     = np.interp(sample_xs, xs_data, ys_data,
                              left=ys_data[0], right=ys_data[-1])

        which = self._draw_which
        if which == 'upper':
            self.upper_bound[label] = bound
        else:
            self.lower_bound[label] = bound

        # Untoggle the active button
        btn_key = 'btn_upper' if which == 'upper' else 'btn_lower'
        rw[btn_key].blockSignals(True)
        rw[btn_key].setChecked(False)
        rw[btn_key].blockSignals(False)
        rw['wf_canvas'].setCursor(Qt.ArrowCursor)

        self._draw_label = None
        self._draw_which = None
        self._draw_pts_x = []
        self._draw_pts_y = []
        self._draw_line  = None

        self._update_draw_status(rw)

        slot = next(i for i, r in enumerate(self.row_widgets) if r['label'] == label)
        self._plot_row(slot)

    def _on_clear_bounds(self, label, slot):
        self.upper_bound[label] = None
        self.lower_bound[label] = None
        if self._draw_label == label:
            self._cancel_active_draw()
        rw = self.row_widgets[slot] if slot < len(self.row_widgets) else None
        if rw:
            self._update_draw_status(rw)
        self._plot_row(slot)

    # =========================================================================
    # Callbacks
    # =========================================================================
    def _on_good_changed(self, label, state):
        self.good_state[label] = bool(state == Qt.Checked)

    def _on_excl_changed(self, label, val, slot):
        self.excl_param[label] = val
        if slot < len(self.row_widgets):
            self._plot_row(slot)

    def _on_isi_param_changed(self):
        self.isi_bin_ms = self.isi_bin_spin.value()
        self.isi_max_ms = self.isi_max_spin.value()
        for rw in self.row_widgets:
            times_f, _ = self._get_filtered(rw['label'])
            self._plot_isi(rw, times_f)

    def _on_wf_nth_changed(self, text):
        try:
            self.waveform_nth = int(text)
        except ValueError:
            return
        for rw in self.row_widgets:
            self._plot_waveform(rw)

    def _on_prev_page(self):
        self.current_page_start = max(0, self.current_page_start - self.units_per_page)
        self._update_nav_buttons()
        self._build_rows()
        self._update_all_plots()
        self._update_range_label()
        self._load_chunk(self.current_page_start)

    def _on_next_page(self):
        n = len(self._visible_labels())
        self.current_page_start = min(
            self.current_page_start + self.units_per_page, max(0, n - 1))
        self._update_nav_buttons()
        self._build_rows()
        self._update_all_plots()
        self._update_range_label()
        self._load_chunk(self.current_page_start)

    # =========================================================================
    # Nav + range label
    # =========================================================================
    def _update_nav_buttons(self):
        loading = (self._chunk_loader is not None and
                   self._chunk_loader.isRunning())
        n     = len(self._visible_labels())
        start = self.current_page_start
        self.btn_prev.setEnabled(start > 0 and not loading)
        self.btn_next.setEnabled(start + self.units_per_page < n and not loading)
        if n == 0:
            self.page_lbl.setText("–")
        else:
            page_num = start // self.units_per_page + 1
            page_tot = max(1, -(-n // self.units_per_page))
            self.page_lbl.setText(f"Page {page_num} of {page_tot}")

    def _update_range_label(self):
        n = len(self._visible_labels())
        if n == 0:
            self.unit_range_lbl.setText("No data loaded")
            return
        lo = self.current_page_start + 1
        hi = min(self.current_page_start + self.units_per_page, n)
        self.unit_range_lbl.setText(f"Showing units {lo}–{hi} of {n}")

    # =========================================================================
    # Export
    # =========================================================================
    def export_good_units(self):
        if not self.unit_labels:
            QMessageBox.warning(self, "No data", "Load spike data first.")
            return

        for rw in self.row_widgets:
            self.good_state[rw['label']] = rw['chk_good'].isChecked()

        good_labels, good_times, good_waves = [], [], []
        for lbl in self.unit_labels:
            if self.good_state.get(lbl, False):
                t_f, w_f = self._get_filtered(lbl)
                good_labels.append(lbl)
                good_times.append(t_f.tolist())
                good_waves.append(w_f.tolist())

        if not good_labels:
            QMessageBox.warning(self, "Nothing to export",
                                "No units are marked as good.")
            return

        session_name = self.loaded_session or "session"
        default_name = f"goodunits_spike_times_dict_{session_name}.pkl"
        default_path = os.path.join(self.loaded_dir, default_name) \
                       if self.loaded_dir else default_name

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save good units", default_path,
            "PKL Files (*.pkl)")
        if not save_path:
            return

        out = {'unit_labels': good_labels,
               'spike_times': good_times,
               'spike_waveforms': good_waves}
        with open(save_path, 'wb') as f:
            pickle.dump(out, f)

        QMessageBox.information(self, "Exported",
            f"Saved {len(good_labels)} good units to:\n{save_path}")


# =============================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = SpikeQualityGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
