from pyroorb.digest2.binning import h_to_bin, qei_to_bin
from pyroorb.digest2.constants import HPART, QPART


def test_qei_to_bin_basic():
    assert qei_to_bin(0.3, 0.05, 1) == (0, 0, 0)
    assert qei_to_bin(99.9, 1.0999, 179.9) == (28, 7, 10)


def test_qei_to_bin_out_of_range():
    assert qei_to_bin(1000.0, 0.05, 1) is None
    assert qei_to_bin(0.3, 5.0, 1) is None
    assert qei_to_bin(0.3, 0.05, 500) is None


def test_qei_to_bin_edges_match_partition_table():
    # a value exactly at a partition edge belongs to the *next* bin (">=" test)
    assert qei_to_bin(QPART[0], 0.05, 1)[0] == 1


def test_h_to_bin_catchall():
    assert h_to_bin(HPART[-1] + 10) == len(HPART) - 1
    assert h_to_bin(0) == 0
    assert h_to_bin(HPART[3]) == 4  # exactly on a boundary -> next bin
