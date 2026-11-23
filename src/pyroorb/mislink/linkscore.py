"""The mislinkage model as an ITF link scorer (``itf.linkscore``, which loads it as a
built-in): it objects to a linkage of 3+ nights whose mislinkage score reaches
:data:`LINK_SCORE_MAX`.

The score must stay below 0.1 on every ITF linkage -- far stricter than the model's 0.93
operating point, which is tuned to flag no correct catalogue arc; a submitted linkage is
held to a much higher bar than a catalogue entry is suspected on.  Out of the model's domain (q < 1.3 au or
a > 5.5 au) the score still objects, saying the objection may be spurious: the model was
trained on main-belt to Hilda arcs.  A linkage the model cannot score at all (outside 3-5
nights, a multi-opposition arc, too few multi-detection nights, a failed fit) gets no
objection -- there is no score to read -- and says so (``itf.linkscore.NO_VERDICT``): a
linkage no search gate judged is then reported unjudged (``itf.srsearch.UNJUDGED``).

The objection is a warning (``itf.srsearch.chimera_warnings``): shown with the linkage, it
blocks only where any warning already blocks (the NEOCP monitor's auto-submit).
"""

#: A linkage scoring at or above this carries the chimera warning.
LINK_SCORE_MAX = 0.1


def objection(obs80_text, obscodes):
    """One sentence when the model scores the linkage ``obs80_text`` at or above
    :data:`LINK_SCORE_MAX`, None when it scores it below, ``NO_VERDICT`` when it cannot
    score it."""
    from pyroorb.itf.linkscore import NO_VERDICT
    from pyroorb.obs_sources import read_obs_file
    from .score import score_observations
    obs, _fmt = read_obs_file(obs80_text, obscodes, in_format="obs80")
    if not obs:
        return NO_VERDICT
    res = score_observations(obs, "LINK", obscodes, explain=False)
    if res.get("status") != "ok" or res.get("score") is None:
        return NO_VERDICT
    s = float(res["score"])
    if s < LINK_SCORE_MAX:
        return None
    # the model's own tier words ("very remote chance" up to 0.27) describe its 0.93
    # operating point, not this cut: quoting them beside a warning contradicts it
    text = (f"the mislinkage model scores this linkage {s:.2f} (a linkage must stay below "
            f"{LINK_SCORE_MAX:g})")
    if not res.get("in_domain"):
        text += (" -- this objection may be spurious: the model was trained on main-belt "
                 "to Hilda arcs (q >= 1.3 au, a <= 5.5 au) and this orbit lies outside them")
    return text


SCORER = objection
