"""The contract between the HUD (gui/hud.html) and the backend behind it.

Both directions are name-addressed across a websocket, so nothing here is checked
by the compiler or by any test that runs only Python: the frontend calls backend
methods by string, and the backend pushes JS function names by string. A rename on
either side produces a dead button or a message that never arrives, silently.

That is exactly what happened when RunHealthCheckTool became SystemReportTool - the
call site in gui/api.py still asked for a key that had moved, the dashboard caught
the error and rendered zeros, and nothing anywhere said so.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HUD = (PROJECT_ROOT / "gui" / "hud.html").read_text()


def _without_comments(text: str) -> str:
    """The file with its comments taken out.

    Several checks below assert that a pattern is ABSENT, and the comments in
    hud.html explain what was removed and why - which means they quote the very
    strings being searched for. Testing the code means testing the code.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)      # /* ... */ and <!-- --> alike
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return re.sub(r"(?<![:'\"])//[^\n]*", " ", text)        # // ... , but not https://


CODE = _without_comments(HUD)
API = (PROJECT_ROOT / "gui" / "api.py").read_text()
SERVER = (PROJECT_ROOT / "gui" / "server.py").read_text()


def _declared_methods():
    sync = set(re.findall(r"['\"]([a-z_]+)['\"]",
                          re.search(r"SYNC_METHODS\s*=\s*\{(.*?)\}", API, re.S).group(1)))
    asyn = set(re.findall(r"['\"]([a-z_]+)['\"]",
                          re.search(r"ASYNC_METHODS\s*=\s*\{(.*?)\}", API, re.S).group(1)))
    return sync, asyn


def _hud_calls():
    return set(re.findall(r"callApi\(\s*['\"]([A-Za-z_]+)['\"]", HUD))


def _backend_pushes():
    sources = API + (PROJECT_ROOT / "core" / "orchestrator.py").read_text()
    return set(re.findall(r'push\(\s*"([A-Za-z]+)"', sources))


def test_every_method_the_hud_calls_is_declared_and_implemented():
    sync, asyn = _declared_methods()
    for method in sorted(_hud_calls()):
        assert method in sync | asyn, f"hud.html calls '{method}', which gui/api.py doesn't declare"
        handler = f"a_{method}" if method in asyn else method
        assert re.search(rf"def {re.escape(handler)}\b", API), f"no {handler}() in gui/api.py"


def test_no_declared_method_is_unreachable_from_the_hud():
    """A method the interface never calls is either a dead endpoint or a button that
    was renamed on one side only."""
    sync, asyn = _declared_methods()
    assert not (sync | asyn) - _hud_calls()


def test_the_server_routes_every_declared_method():
    sync, asyn = _declared_methods()
    dispatch = re.search(r"async def _dispatch_call.*?(?=\n    async def |\n    def )",
                         SERVER, re.S).group(0)
    for method in sorted(asyn):
        assert f'"{method}"' in dispatch, f"gui/server.py has no branch for async method '{method}'"
    assert "SYNC_METHODS" in dispatch, "sync methods are routed as a group; that route is gone"


def test_every_push_is_allowlisted_and_defined_in_the_page():
    """The HUD dispatches pushes through an allowlist, so a backend push the page
    doesn't know about is dropped with a console warning the user never sees."""
    allowed = set(re.findall(r"['\"]([a-zA-Z]+)['\"]",
                             re.search(r"PUSH_HANDLERS = new Set\(\[(.*?)\]\)", HUD, re.S).group(1)))
    defined = set(re.findall(r"window\.([A-Za-z]+)\s*=\s*function", HUD))
    for fn in sorted(_backend_pushes()):
        assert fn in allowed, f"backend pushes '{fn}', which hud.html's allowlist drops"
        assert fn in defined, f"hud.html allowlists '{fn}' but defines no window.{fn}"


def test_no_allowlisted_push_is_dead():
    assert not set(re.findall(r"['\"]([a-zA-Z]+)['\"]",
                              re.search(r"PUSH_HANDLERS = new Set\(\[(.*?)\]\)", HUD, re.S).group(1))
                   ) - _backend_pushes()


def test_the_dashboard_asks_system_report_for_the_section_it_reads():
    """gui/api.py reads output['health']['metrics']; if it stops asking for the
    health section, or the section is renamed, the dashboard reads a key that isn't
    there and reports zeros rather than failing visibly."""
    stats = re.search(r"async def a_get_system_stats.*?(?=\n    async def |\n    def )",
                      API, re.S).group(0)
    assert 'sections=["health"]' in stats
    assert '"health"' in stats and '"metrics"' in stats


# --- The elements the interface's own script reaches for -------------------------
# The HUD is one HTML file: markup and behaviour are only connected by id, so a
# renamed or dropped element is a silent dead control, not an error. These pin the
# ids the script binds to - the ones a redesign is most likely to lose.

def _element_ids():
    return set(re.findall(r"""id=["']([A-Za-z][A-Za-z0-9_-]*)["']""", HUD))


def test_every_id_the_script_binds_to_exists_in_the_markup():
    referenced = set(re.findall(r"""getElementById\(\s*['"]([A-Za-z][A-Za-z0-9_-]*)['"]""", HUD))
    # Ids the script CREATES and then binds to (the audio card and settings form
    # build their own markup), so they legitimately aren't in the static page.
    built_at_runtime = {
        "audioDevice", "audioTestMic", "audioTestSpk", "audioLevel", "audioSpkDetail",
        "audioDismiss", "audioAllow", "audioDecline", "audioSaveNote", "spkYes", "spkNo",
        "emptyHint",
    }
    missing = sorted(referenced - _element_ids() - built_at_runtime)
    assert not missing, f"the script binds to ids that no longer exist: {missing}"


def test_the_features_that_have_to_survive_a_redesign_are_all_present():
    """Each of these is a working capability behind a piece of the interface. A
    restyle that drops one leaves the backend method wired to nothing."""
    required = {
        "clockTime": "the top-bar clock",
        "clockDate": "the date",
        "weatherTemp": "weather",
        "weatherDesc": "weather description",
        "cpuVal": "CPU readout", "memVal": "memory readout", "diskVal": "disk readout",
        "cpuFill": "CPU bar", "memFill": "memory bar", "diskFill": "disk bar",
        "todoInput": "adding a task", "todoAddBtn": "the add button", "todoList": "the task list",
        "chatInput": "typing to Leti", "sendBtn": "sending",
        "sessionPanel": "the session log", "sessionPanelInner": "the log body",
        "historyBtn": "expanding the log",
        "micBtn": "the microphone toggle", "simBtn": "the speaking preview",
        "orbPath": "the audio-reactive core", "stateLabel": "the state readout",
        "audioOverlay": "the first-run audio card", "audioContent": "its body",
        "audioClose": "closing it", "audioBtn": "re-opening it",
        "settingsOverlay": "the settings modal", "settingsContent": "its body",
        "settingsClose": "closing it",
        "lightbox": "full-size images", "lightboxImg": "the image itself",
    }
    ids = _element_ids()
    missing = {k: v for k, v in required.items() if k not in ids}
    assert not missing, f"the redesign dropped: {missing}"


def test_the_audio_card_can_always_be_reopened():
    """blocked_reason() tells the user to 'press Audio in the interface'. That
    promise needs something in the interface that opens it."""
    openers = re.findall(r"getElementById\('([A-Za-z]+)'\)\.onclick = \(\) => openAudioSetup", HUD)
    assert openers, "nothing in the HUD re-opens the audio setup card"


def test_the_settings_modal_can_be_opened_without_typing_a_command():
    """/settings still works in the chat box, but a pointer-only client (a phone)
    needs a control too."""
    assert re.search(r"getElementById\('[A-Za-z]+'\)\.onclick = \(\) => openSettingsModal", HUD)


# --- The radar animation ----------------------------------------------------------
# Smoothness here is a property of how the loop is written, and each of these
# pins one thing that was actually wrong before: motion measured in frames rather
# than seconds, a fresh loop per mode change, per-frame randomness, and four
# separate timelines that resumed differently after a hidden tab.

def test_motion_is_measured_in_seconds_not_frames():
    """`t += 0.02` per frame ran at double speed on a 120Hz screen and slowed down
    whenever frames were dropped. Every moving thing has to scale by elapsed time."""
    assert "const dt = Math.min(" in HUD, "no clamped delta-time in the animation loop"
    assert re.search(r"clock \+= dt", HUD), "the animation clock is not advanced by elapsed time"
    # The old per-frame counters are gone.
    assert "t += 0.02" not in CODE and "t += 0.35" not in CODE and "t += 0.05" not in CODE


def test_there_is_exactly_one_animation_loop():
    """Four loops that each cancelled the others is what made a mode change jump:
    the new one started from its own counter."""
    starts = re.findall(r"requestAnimationFrame\((\w+)\)", CODE)
    assert set(starts) <= {"tickFrame", "r"}, f"more than one animation loop drives the radar: {set(starts)}"
    for dead in ("idleFrame", "speakingFrame", "listeningFrame", "listeningIdleFrame"):
        assert dead not in CODE, f"{dead} is a leftover second loop"


def test_the_drawn_shape_eases_towards_its_target():
    """Frame-rate independent smoothing - the fraction covered depends on elapsed
    time, not on how many frames fitted into it. This is what makes 60Hz and
    144Hz look the same rather than merely run at the same speed."""
    assert "Math.exp(-dt /" in HUD


def test_the_speaking_animation_is_not_re_randomised_every_frame():
    """Math.random() per bin per frame is not a wobble, it is sixty unrelated
    shapes a second."""
    speaking = re.search(r"if\(mode === 'speaking'\)\{.*?\n      return;", CODE, re.S)
    assert speaking, "the speaking branch moved; this test needs updating"
    assert "Math.random()" not in speaking.group(0), "speaking still randomises per frame"
    assert "noiseAt(" in speaking.group(0), "speaking should read interpolated noise"


def test_the_radar_has_no_smil_timelines_left():
    """SMIL animations run on their own clocks beside the script's, and a
    backgrounded tab brings them back out of step with it."""
    assert "<animateTransform" not in CODE
    assert "<animate " not in CODE


def test_a_hidden_tab_does_not_come_back_with_one_enormous_frame():
    assert "visibilitychange" in HUD
    assert "document.hidden" in HUD


def test_reduced_motion_is_respected():
    """A spinning radar is not something to hand someone who asked their system
    for less motion. The core still tracks audio - that part is information."""
    assert "prefers-reduced-motion" in HUD
    assert "REDUCED_MOTION" in HUD


def test_the_meter_is_one_path_not_twenty_eight_elements():
    """The meter used to be 28 <i> elements whose inline heights were rewritten
    every frame, which made the browser lay the whole page out sixty times a
    second for a decoration. One path draws the same bars and changes nothing
    about layout. (Measured in Chromium: layout duration fell about a quarter.)

    Also guards the older bug: a CSS transition on top of a per-frame update is a
    second interpolation fighting the first, which is what made the bars lag.
    """
    assert "waveformPath" in CODE
    assert 'setAttribute(\'d\'' in CODE or "setAttribute('d'" in CODE
    rule = re.search(r"\.waveform\{(.*?)\}", CODE, re.S)
    assert rule and "transition" not in rule.group(1)


def test_the_slow_rings_are_not_redrawn_for_movement_nobody_can_see():
    """They turn at 6 and 4 degrees a second. At 60fps that is a sixteenth of a
    degree per frame - under a pixel at that radius - and on the tick ring it
    repaints forty-eight lines to achieve it."""
    assert "RING_STEP_DEGREES" in CODE
    assert "lastRingUpdate" in CODE


def test_the_idle_redraw_rate_is_limited_by_not_asking_for_frames():
    """Asking for a frame is the expensive part, not what happens inside one: the
    browser runs its whole frame pipeline per request. An earlier attempt asked for
    every frame and skipped the drawing in most of them, which measured as no
    saving at all. The rate is now limited by scheduling."""
    assert "IDLE_REDRAW_FPS" in CODE
    schedule = re.search(r"function scheduleNextFrame\(\)\{(.*?)\n  \}", CODE, re.S)
    assert schedule, "scheduleNextFrame moved; this test needs updating"
    body = schedule.group(1)
    assert "setTimeout" in body and "IDLE_REDRAW_FPS" in body
    assert "requestAnimationFrame(tickFrame)" in body, "no full-rate path for live audio"


def test_live_audio_is_never_throttled():
    """Idle, the only motion is a breath and a slow sweep. Following a microphone,
    the frame IS the information, so the cap must not apply there."""
    schedule = re.search(r"function scheduleNextFrame\(\)\{(.*?)\n  \}", CODE, re.S)
    assert "if(mode === 'idle')" in schedule.group(1)


def test_leaving_idle_does_not_wait_out_the_idle_timer():
    mode_fn = re.search(r"function setMode\(newMode\)\{(.*?)\n  \}", CODE, re.S)
    assert mode_fn and "clearTimeout(idleTimer)" in mode_fn.group(1)


def test_stopping_the_loop_clears_a_pending_frame():
    """A pending timer would wake the loop again after it was told to stop - that is
    how a 'paused' animation keeps burning CPU behind a hidden window."""
    stop = re.search(r"function stopAnimation\(\)\{(.*?)\n  \}", CODE, re.S)
    assert stop and "clearTimeout(idleTimer)" in stop.group(1)


def test_there_are_no_perpetual_css_animations():
    """A CSS animation that never ends keeps the browser's whole frame pipeline
    running for as long as the page is open. The one that existed here - a pulsing
    7px status dot - measured at roughly 1.7 cores on its own in the desktop
    window's renderer, the single largest cost in the application. Promoting it to
    its own layer did not help; it is the running animation itself that costs.
    Anything that needs to pulse goes on the animation loop's clock instead."""
    assert "infinite" not in CODE, "a perpetual CSS animation is back"
    assert "@keyframes" not in CODE
    # ...and the dot is still driven, just from the loop.
    assert "statusDot.style.opacity" in CODE


def test_the_status_dot_has_a_layer_of_its_own():
    """The loop writes this dot's opacity on every drawn frame. Without a layer,
    each write repaints the dot, its 8px glow and the panel behind them; with one,
    the same write composites and paints nothing. Measured in the desktop window's
    renderer at 214.7% of a core down to 203.3% - the whole cost of the write."""
    dot = re.search(r"\.status-dot\{(.*?)\}", CODE, re.S)
    assert dot, ".status-dot moved; this test needs updating"
    assert "will-change:opacity" in dot.group(1).replace(" ", "")


def test_the_overlays_are_promoted_only_where_that_was_measured_to_help():
    """Giving the two full-viewport overlays their own layers takes a third off the
    page's CPU in Blink and ADDS a seventh in the WebKitGTK build behind the native
    window, which composites layers on the CPU. So it is a class, applied to the
    browsers and phones it helps and withheld from the windows it hurts - not a
    blanket declaration picked for one engine and inflicted on the other."""
    assert ".promote-overlays #leti-root::before" in CODE
    assert ".promote-overlays #leti-root::after" in CODE
    # The bare pseudo-elements must NOT carry it themselves.
    for pseudo in ("#leti-root::before", "#leti-root::after"):
        rule = re.search(re.escape("\n  " + pseudo + "{") + r"(.*?)\}", CODE, re.S)
        assert rule, f"{pseudo} moved; this test needs updating"
        assert "will-change" not in rule.group(1), (
            f"{pseudo} is promoted unconditionally, which regresses the native window")


def test_the_promotion_is_decided_by_the_engine_not_the_window():
    """Blink gains from the promotion and WebKit loses by it, so the engine is what
    the decision has to key off. Keying it off "is this a native window" was right
    only on Linux, where that window happens to be WebKit: on Windows it is Edge
    WebView2, which is Blink, and the rule withheld the saving from exactly the
    machines it would have helped. overflow-anchor separates the two (Blink and
    Gecko support it, WebKit does not), and it is checked in the pre-paint block so
    no already-composited page gets re-layered."""
    pre = CODE.split("<style>")[0]
    assert "promote-overlays" in pre, "the decision is being made after the first paint"
    assert "overflow-anchor" in pre, "the promotion is not keyed off the engine"
    assert "desktop=1" not in CODE, (
        "the promotion is keyed off the window again, which is wrong on Windows")


def test_the_pre_paint_block_leaks_no_globals():
    """Top-level `const` in a classic script shares the global lexical scope with
    the main script below, where a collision is a SyntaxError that takes the whole
    page down rather than shadowing a variable."""
    pre = CODE.split("<style>")[0]
    block = pre[pre.index("URLSearchParams") - 400:]
    assert re.search(r"\{\s*const params = new URLSearchParams", block), (
        "the pre-paint block's declarations are no longer scoped to a block")


# --- Minimised mode ---------------------------------------------------------------

def test_minimising_is_a_class_not_a_teardown():
    """The puck has to be the same radar, still driven by the same loop and the
    same socket - a session that shrank, not one that stopped."""
    ids = _element_ids()
    assert "minimizeBtn" in ids and "puckBadge" in ids
    assert ".minimized" in HUD
    # No second SVG, no second renderer.
    assert HUD.count('id="orbPath"') == 1
    assert HUD.count('id="sweepGroup"') == 1


def test_the_puck_says_who_it_is_and_what_it_is_doing():
    assert 'class="puck-name"' in HUD
    assert "#leti-root.minimized .puck-name{ display:block; }" in HUD
    # The state readout is shared with the full-size view, so it keeps updating.
    assert "#leti-root.minimized .state{" in HUD


def test_replies_arriving_while_minimised_are_counted():
    """Minimised, the log is off screen; without the badge an answer would arrive
    with nothing at all to show for it."""
    reply = re.search(r"window\.appendLetiReply = function.*?\n  \};", HUD, re.S)
    assert reply and "unseen" in reply.group(0)


def test_the_puck_can_always_be_reopened():
    assert re.search(r"radarEl\.addEventListener\('pointerup'", CODE)
    assert "restore()" in CODE
    assert "Escape" in CODE


def test_dragging_the_puck_does_not_also_reopen_it():
    """The puck is both the thing you move and the thing you click. Without a
    movement threshold every drag would end by reopening the window it was being
    dragged out of the way."""
    assert "DRAG_SLOP" in CODE
    end = re.search(r"function endDrag\(e\)\{(.*?)\n  \}", CODE, re.S)
    assert end, "endDrag moved; this test needs updating"
    assert "if(!wasDrag && minimized) restore();" in end.group(1)


def test_the_puck_asks_for_a_real_window_before_settling_for_a_smaller_layout():
    """Overlapping other applications needs a native always-on-top window. In a
    browser tab there isn't one, so the same control collapses the layout instead
    - but it must try for the real thing first."""
    assert "callApi('set_window_mode', 'puck')" in CODE
    minimize = re.search(r"async function minimize\(\)\{(.*?)\n  \}", CODE, re.S)
    assert minimize and "if(!swapped) applyMinimizedLayout(true);" in minimize.group(1)


def test_only_the_native_window_asks_to_swap_windows():
    """Every client shares one session, so a phone minimising must not hide the
    desktop app's windows on someone else's screen. The window a client can
    shrink is the window it is."""
    minimize = re.search(r"async function minimize\(\)\{(.*?)\n  \}", CODE, re.S)
    assert minimize, "minimize() moved; this test needs updating"
    body = minimize.group(1)
    guard = body.index("IN_DESKTOP_WINDOW")
    call = body.index("set_window_mode")
    assert guard < call, "the window swap is not gated on actually being a native window"


def test_the_puck_window_renders_itself_collapsed():
    """The small window loads the same page with ?puck=1 rather than a second,
    smaller interface that would have to be kept in step with the first."""
    assert "IS_PUCK_WINDOW" in CODE
    assert "?puck=1" in CODE or "'puck'" in CODE
    assert HUD.count('id="orbPath"') == 1, "the puck must not be a second renderer"


def test_the_puck_window_is_transparent_behind_the_circle():
    assert "html.puck-window, html.puck-window body{ background:transparent" in HUD


def test_native_dragging_is_scoped_to_a_region_the_page_marks():
    """easy_drag moves the window on any drag anywhere, which swallows the click
    that reopens the interface."""
    assert "pywebview-drag-region" in CODE
    assert "pywebviewready" in CODE


def test_confirmations_still_reach_the_user_while_minimised():
    """A modify action can ask for confirmation at any moment, including while the
    interface is a coaster in the corner. If the overlay can't be clicked, that
    action is unanswerable."""
    assert "#leti-root.minimized .settings-overlay" in HUD


# --- Weather offline --------------------------------------------------------------

def test_a_failed_refresh_does_not_blank_a_good_reading():
    """Losing the backend for one poll should age the number, not erase it."""
    assert "haveReading" in HUD
    unavailable = re.search(r"function setWeatherUnavailable\(\)\{(.*?)\n  \}", HUD, re.S)
    assert unavailable and "if(haveReading) return;" in unavailable.group(1)


def test_a_cached_reading_is_labelled_as_one_in_the_interface():
    """The backend marks it stale; the panel has to actually say so."""
    assert "weatherAge" in _element_ids()
    assert "data.stale" in HUD and "describeAge" in HUD


# --- Load-time ordering inside the one big script ---------------------------------

def test_the_minimised_layout_is_applied_after_what_it_touches():
    """A real bug, and one a static check is the right shape for.

    The puck window applies its collapsed layout as the page loads, and that call
    reaches into the session panel. While the minimise block sat ABOVE the session
    panel's `let sessionExpanded`, that load-time call hit the temporal dead zone
    and threw - after the CSS class had already been set. So the page LOOKED
    collapsed while every line below the throw never ran, which included the five
    window.* push handlers: the puck rendered perfectly and could not receive a
    single reply.

    The earlier test asserted the class was present and passed happily.
    """
    declaration = HUD.index("let sessionExpanded")
    block = HUD.index("// ---------- Minimised mode ----------")
    assert declaration < block, (
        "the minimise block runs setSessionExpanded() at load; moving it above that "
        "declaration puts the call in the temporal dead zone and kills the rest of "
        "the script"
    )


def test_the_push_handlers_are_defined_before_the_minimise_block_runs():
    """Same failure from the other side: appendLetiReply has to exist whatever
    happens later in the file, because a push can arrive at any time."""
    assert HUD.index("window.appendLetiReply") < HUD.index("// ---------- Minimised mode ----------")


def test_an_unfocused_window_redraws_less():
    """The interface sits open all day. While it is behind whatever the user is
    actually working in, redrawing a decorative sweep for an audience that is not
    looking is exactly the kind of work that should not be happening - and unlike
    a lower rate while focused, nobody can see the difference."""
    assert "UNFOCUSED_REDRAW_FPS" in CODE
    schedule = re.search(r"function scheduleNextFrame\(\)\{(.*?)\n  \}", CODE, re.S)
    assert schedule and "windowFocused" in schedule.group(1)
    assert "addEventListener('blur'" in CODE and "addEventListener('focus'" in CODE


def test_refocusing_does_not_wait_out_the_slow_interval():
    focus = re.search(r"window\.addEventListener\('focus', \(\) => \{(.*?)\n  \}\);", CODE, re.S)
    assert focus and "clearTimeout(idleTimer)" in focus.group(1)


def test_the_three_idle_states_are_all_handled():
    """Hidden, unfocused and focused are different amounts of nobody-is-looking,
    and each gets its own answer: no frames, few frames, and a smooth rate."""
    assert "stopAnimation()" in CODE and "document.hidden" in CODE   # hidden: none
    assert "UNFOCUSED_REDRAW_FPS" in CODE                            # unfocused: few
    assert "IDLE_REDRAW_FPS" in CODE                                 # focused idle
