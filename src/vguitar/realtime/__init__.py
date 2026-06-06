"""Realtime layer: live audio I/O, RTF measurement, and an offline renderer.

This subpackage is where a :class:`~vguitar.models.base.Model` meets a clock.
Three entry points, in increasing order of "needs hardware":

* :func:`measure_rtf` — no audio device; times ``process_block`` to get the
  real-time factor (audio seconds produced per compute second).
* :func:`render_file` — no audio device; streams a wav through the model
  block-by-block so you can *listen* to the result offline.
* :class:`LiveEngine` — opens a duplex PortAudio stream and runs the model in
  the callback for true live monitoring.

See :mod:`vguitar.realtime.engine` for the realtime-safety notes.
"""

from __future__ import annotations

from vguitar.realtime.engine import LiveEngine, measure_rtf, render_file

__all__ = ["LiveEngine", "measure_rtf", "render_file"]
