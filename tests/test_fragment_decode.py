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


def test_slicing_the_predictor_does_not_change_the_output():
    """Predicting in slices must be a MEMORY optimisation and nothing else.

    `predict_fragment_batches` calls the model in slices of `predict_chunk` keys so peak memory is
    one slice rather than the whole digest. That is only safe if the slice boundary is invisible in
    the output: same rows, same order, same values. A slice size that does not divide the key count
    is the case that would expose an off-by-one at the boundary, so the small chunk here is chosen
    NOT to divide it.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from timsim_predict import fragments as F

    # 7 distinct (sequence, charge) KEYS, each shared by 3 precursors -- so the fanout through
    # key2pids is actually exercised. Pairing 7 sequences with charges [2, 3] would give FOURTEEN
    # keys of one precursor each and prove nothing about fanout, which is what an earlier version of
    # this test did while claiming otherwise.
    seqs = [f"PEPTIDEK{i}" for i in range(7)]
    rows = [(s, 2) for s in seqs for _ in range(3)]        # 7 keys x 3 precursors = 21 rows
    prec = pd.DataFrame({
        "precursor_id": np.arange(len(rows), dtype=np.uint64),
        "sequence": [r[0] for r in rows],
        "charge": [r[1] for r in rows],
    })
    n_keys = len({(a, b) for a, b in rows})
    assert n_keys == 7, f"fixture must have 7 keys to straddle chunk=3, got {n_keys}"

    rng = np.random.default_rng(0)
    table = {}

    seen_calls = []

    def fake_predict(sequences, charges, collision_energies, model=None):
        """Deterministic per (sequence, charge, CE), so a slice cannot change what a key predicts.

        CE is folded into the key on purpose: if a slice ever handed the model a CE array misaligned
        with its sequences, the tensors would differ and the comparison would catch it. A fake keyed
        only on (sequence, charge) would be blind to exactly that bug.
        """
        seqs_l, chg_l, ce_l = list(sequences), list(charges), list(collision_energies)
        seen_calls.append([(str(s), int(c), round(float(e), 6))
                           for s, c, e in zip(seqs_l, chg_l, ce_l)])
        out = np.zeros((len(seqs_l), 29, 2, 3), dtype=np.float32)
        for i, (s, c, e) in enumerate(zip(seqs_l, chg_l, ce_l)):
            key = (str(s), int(c), round(float(e), 6))
            if key not in table:
                table[key] = rng.random((29, 2, 3)).astype(np.float32)
            out[i] = table[key]
        return out, "fake-model"

    original = F.predict_tensors
    F.predict_tensors = fake_predict
    try:
        def run(predict_chunk, frame=None):
            _prov, schema, batches = F.predict_fragment_batches(
                frame if frame is not None else prec, 25.0, model="fake-model",
                verbose=False, predict_chunk=predict_chunk)
            return pa.Table.from_batches(list(batches), schema=schema)

        whole = run(10_000)      # one slice: the pre-existing behaviour
        sliced = run(3)          # three slices, the last one short
        # Same again through the PER-PRECURSOR collision-energy branch, which resolves a CE per key
        # and slices that array alongside the sequences. It has its own indexing and was previously
        # untested.
        # CE must VARY BETWEEN keys (and stay constant within one, which the CE resolver enforces).
        # A single constant CE would make every slicing of the CE array identical, so a misaligned
        # slice would be invisible — which is exactly what an earlier version of this test did.
        ce_of = {s_: 20.0 + 3.0 * i for i, s_ in enumerate(seqs)}
        prec_ce = prec.assign(collision_energy=prec["sequence"].map(ce_of))
        seen_calls.clear()
        whole_ce = run(10_000, prec_ce)
        calls_whole = [c for c in seen_calls]
        seen_calls.clear()
        sliced_ce = run(3, prec_ce)
        calls_sliced = [c for c in seen_calls]
    finally:
        F.predict_tensors = original

    assert whole.num_rows > 0, "the fixture produced no fragments; the test would prove nothing"
    # Fanout really happened: 21 precursors over 7 keys, and every precursor got rows.
    assert whole.column("precursor_id").to_pylist(), "no precursor ids in the output"
    assert len(set(whole.column("precursor_id").to_pylist())) == 21, \
        "expected all 21 precursors to receive fragments — key2pids fanout is not being exercised"
    assert whole.equals(sliced), "slicing changed the output"
    assert whole_ce.equals(sliced_ce), "slicing changed the output on the per-precursor CE branch"
    # And the model saw the SAME (sequence, charge, CE) triples in the same order either way --
    # equal output tables could in principle hide compensating misalignment.
    assert [t for c in calls_whole for t in c] == [t for c in calls_sliced for t in c], \
        "slicing handed the model different (sequence, charge, CE) pairings"
