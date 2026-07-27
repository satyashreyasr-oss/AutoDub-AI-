from dubbing.synthesizer import MAX_TEMPO, MIN_TEMPO, compute_tempo


def test_matching_durations_need_no_stretch():
    assert compute_tempo(raw_duration=10.0, target_duration=10.0) == 1.0


def test_tts_longer_than_slot_speeds_up():
    # 20s of speech into a 10s slot needs 2x tempo, clamped to MAX_TEMPO.
    assert compute_tempo(raw_duration=20.0, target_duration=10.0) == MAX_TEMPO


def test_tts_shorter_than_slot_slows_down():
    # 5s of speech into a 10s slot needs 0.5x tempo, clamped to MIN_TEMPO.
    assert compute_tempo(raw_duration=5.0, target_duration=10.0) == MIN_TEMPO


def test_moderate_mismatch_is_not_clamped():
    # 12s of speech into a 10s slot needs 1.2x, within [MIN_TEMPO, MAX_TEMPO].
    assert compute_tempo(raw_duration=12.0, target_duration=10.0) == 1.2


def test_zero_length_target_does_not_divide_by_zero():
    # A zero/negative-duration segment must floor to a small slot rather than
    # divide by zero; any real speech into that tiny slot clamps to MAX_TEMPO.
    tempo = compute_tempo(raw_duration=1.0, target_duration=0.0)
    assert tempo == MAX_TEMPO
