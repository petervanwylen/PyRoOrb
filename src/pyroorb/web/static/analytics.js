// Google Analytics events for the PyRoOrb web UI. Vanilla JS, no build step;
// loaded BEFORE app.js (the gtag stub itself is the inline block in index.html).
//
// One entry point, gaEvent(name, params), so every call site reads the same and
// none of them has to care whether analytics is there at all: an ad-blocker
// that drops gtag.js, a page served without the tag, or a jsdom test that never
// loads this file must all leave the UI exactly as it was. Hence the guards --
// a tracking call may cost nothing and may never throw into the code it rides on.
//
// Events (GA4 names: snake_case, <= 40 chars):
//   fit_object                 a designation fit was started {source}
//   paste_observations         pasted astrometry fitted {action: fit | add | neocp_fit, n_lines}
//   initiate_itf_search        ITF search started      {mag_limit, rank_mode}
//   itf_search_complete        ITF search answered     {outcome, n_tracklets, added_obs, elapsed_sec}
//   switch_to_<pane>_mode      central pane picked     {pane}  (obs / orbarium / ephem / impact)
//   download_linkage_file      linkage JSON downloaded
//   submit_linkage             linkage sent to the MPC {accepted}
//   export                     Export menu product     {export_type}
//   get_random_object          random object drawn     {family}
//   fullscreen_orbit           3D orbit view went fullscreen
//   change_force_model         USER changed the force model {control: nongrav | asteroids, model}
//                              (sent from the controls' listeners, never from a re-fit)
// Parameters carry no observer identity -- never the linkage form's name/e-mail.
function gaEvent(name, params) {
  try {
    if (typeof gtag === "function") gtag("event", name, params || {});
  } catch (e) { /* analytics must never break the page */ }
}
