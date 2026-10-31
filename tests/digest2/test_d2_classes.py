from pyroorb.digest2.classes import CLASS_ABBR, CLASS_TESTS, N_CLASSES, class_index


def test_class_table_lengths():
    assert N_CLASSES == 15
    assert len(CLASS_ABBR) == N_CLASSES
    assert len(CLASS_TESTS) == N_CLASSES


def test_class_index_lookup():
    assert class_index("NEO") == 1
    assert class_index("NEO(q < 1.3)") == 1
    assert class_index("MB1") == 7
    assert class_index("not-a-class") is None


def test_neo_is_just_q_threshold():
    is_neo = CLASS_TESTS[1]
    assert is_neo(1.29, 0.5, 10, 20) is True
    assert is_neo(1.31, 0.5, 10, 20) is False


def test_h18_neo_requires_neo_and_bright():
    is_h18 = CLASS_TESTS[3]
    assert is_h18(1.0, 0.1, 5, 18.4) is True
    assert is_h18(1.0, 0.1, 5, 18.6) is False
    assert is_h18(2.0, 0.1, 5, 10.0) is False  # not a NEO at all


def test_main_belt_classes_are_mutually_exclusive_by_inclination():
    # inner MB caps inclination as a function of semimajor axis; a very
    # inclined orbit at typical inner-MB (a,e) should not qualify.
    is_inner_mb = CLASS_TESTS[7]
    assert is_inner_mb(2.2, 0.0, 5.0, 15) is True
    assert is_inner_mb(2.2, 0.0, 30.0, 15) is False
