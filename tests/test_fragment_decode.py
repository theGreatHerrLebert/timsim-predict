"""Pin the Prosit ``(29, 2, 3)`` axis-2 convention against pepdl's own flattener.

``fragments._AXIS2_ION`` asserts that axis-2 index 0 is a *y* ion and index 1 is a *b* ion. That
fact is taken from ``pepdl._vendored.flatten_prosit_array``'s source, where the charge-major flat-174
layout is written as ``[y_c1(29), b_c1(29), y_c2(29), b_c2(29), y_c3(29), b_c3(29)]`` — i.e.
``array[:, 0, c]`` is y and ``array[:, 1, c]`` is b.

If either side ever moves, this test fails instead of every predicted spectrum silently transposing.
"""

import numpy as np
import pytest

from timsim_predict.fragments import _AXIS2_ION, decode_tensor


def test_axis2_convention_matches_flatten_prosit_array():
    """Probe each (ion type, charge) slot and read back where the flattener puts it."""
    flatten = pytest.importorskip("pepdl._vendored").flatten_prosit_array

    for t, expected_ion in _AXIS2_ION.items():
        for c in range(3):
            probe = np.zeros((29, 2, 3), dtype=np.float64)
            probe[0, t, c] = 1.0
            flat = np.asarray(flatten(probe))
            (hit,) = np.flatnonzero(flat)
            # Charge-major blocks of 29, two blocks (y then b) per charge.
            block = hit // 29
            assert block // 2 == c, f"charge block moved for axis2={t}, c={c}"
            ion_in_block = "y" if block % 2 == 0 else "b"
            assert ion_in_block == expected_ion, (
                f"axis-2 index {t} flattens as {ion_in_block!r}, but _AXIS2_ION says "
                f"{expected_ion!r} — the Prosit tensor convention changed"
            )


def test_decode_tensor_reports_the_pinned_slots():
    """decode_tensor must report (ion, ordinal, charge) matching the tensor indices it read."""
    pred = np.full((29, 2, 3), -1.0, dtype=np.float32)  # Prosit marks absent slots with -1
    pred[0, 0, 0] = 0.5   # y1, +1
    pred[4, 1, 2] = 0.25  # b5, +3
    pred[7, 0, 1] = 1e-6  # below floor -> dropped

    got = sorted(decode_tensor(pred, floor=1e-3))
    assert got == [("b", 5, 3, pytest.approx(0.25)), ("y", 1, 1, pytest.approx(0.5))]
