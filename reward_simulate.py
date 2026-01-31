# reward_simulate.py
# Terminal minimap viewer for ImmortalSufferingEnv with manual MultiDiscrete([2]*8) actions.
# - Reads obs via parse_observation()
# - Builds id_map by robust color encoding (exact + nearest fallback)
# - Renders id_map with curses using palette colors
# - Keyboard:
#   1-8 : toggle action bits
#   space: step once
#   a    : toggle auto-run
#   r    : random action
#   0    : clear action bits
#   q    : quit
#   t    : toggle encoding mode (exact_only / exact+nearest / nearest_only)
#   c    : toggle channel mode (auto / rgb / bgr)
#   m    : toggle downsample (mode / priority)
#   [ / ]: decrease/increase nearest tolerance (L1). 0 means "always nearest" (no threshold)

import time
import argparse
import numpy as np
import curses
from collections import Counter

from libimmortal.env import ImmortalSufferingEnv

# --- Fix these imports to match your project structure (you already had them) ---
from libimmortal.utils.aux_func import parse_observation
from libimmortal.utils.aux_func import DEFAULT_ENCODER
# -----------------------------------------------------------------------------


# Vector indices (as you stated)
IDX_CUM_DAMAGE = 4
IDX_IS_ACTIONABLE = 5
IDX_GOAL_DIST = 11
IDX_TIME = 12


def infer_done(done, info):
    if isinstance(done, (bool, np.bool_)):
        return bool(done)
    if info is not None and info.__class__.__name__ == "TerminalSteps":
        return True
    return False


# -----------------------------
# Robust image preprocessing
# -----------------------------
def to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert to (H,W,3) uint8 robustly.
    Handles:
      - (3,H,W) or (H,W,3)
      - float in [0,1] or [0,255]
      - uint8 but accidentally in {0,1}
    """
    if img is None:
        raise ValueError("graphic_obs is None")

    arr = np.asarray(img)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape={arr.shape}")

    # CHW -> HWC if needed
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3, got shape={arr.shape}")

    # Convert dtype / scale
    if arr.dtype == np.uint8:
        mx = int(arr.max()) if arr.size else 0
        # If it's only 0/1, assume it came from float[0,1] cast to uint8
        if mx <= 1:
            arr = (arr.astype(np.float32) * 255.0).round().clip(0, 255).astype(np.uint8)
        return arr

    # float or other ints
    if np.issubdtype(arr.dtype, np.floating):
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx <= 1.0 + 1e-6:
            arr = (arr * 255.0).round()
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return arr

    # int16/int32/etc
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def maybe_bgr(arr_hwc: np.ndarray, channel_mode: str) -> np.ndarray:
    """
    channel_mode: 'rgb', 'bgr', 'auto'
    """
    if channel_mode == "rgb":
        return arr_hwc
    if channel_mode == "bgr":
        return arr_hwc[..., ::-1]
    if channel_mode == "auto":
        # decide later externally (we keep this function simple)
        return arr_hwc
    raise ValueError(f"Unknown channel_mode={channel_mode}")


# -----------------------------
# Encoding: exact + nearest
# -----------------------------
def exact_match_ratio(img_hwc_u8: np.ndarray, palette_u8: np.ndarray) -> float:
    """
    Returns ratio of pixels that match exactly one of palette colors.
    """
    H, W, _ = img_hwc_u8.shape
    pix = img_hwc_u8.reshape(-1, 3)
    pal = palette_u8.reshape(1, -1, 3)
    # (N,K,3) -> (N,K)
    matches = (pix[:, None, :] == pal).all(axis=-1)
    matched = matches.any(axis=1)
    return float(matched.mean())


def encode_ids_exact(img_hwc_u8: np.ndarray, encoder) -> np.ndarray:
    """
    Use exact palette matching only.
    Unmatched -> encoder.unknown_id
    """
    palette = np.asarray(encoder.palette, dtype=np.uint8)  # (K,3)
    H, W, _ = img_hwc_u8.shape
    pix = img_hwc_u8.reshape(-1, 3)
    pal = palette.reshape(1, -1, 3)

    matches = (pix[:, None, :] == pal).all(axis=-1)  # (N,K)
    matched_any = matches.any(axis=1)
    ids = matches.argmax(axis=1).astype(np.uint8)
    if not matched_any.all():
        ids[~matched_any] = np.uint8(encoder.unknown_id)
    return ids.reshape(H, W)


def encode_ids_nearest(img_hwc_u8: np.ndarray, encoder, tol_l1: int = 0) -> np.ndarray:
    """
    Nearest-color encoding in L1 distance.
    If tol_l1 > 0, pixels with min L1 distance > tol_l1 become unknown_id.
    If tol_l1 == 0, always assign nearest palette color (no threshold).
    """
    palette = np.asarray(encoder.palette, dtype=np.uint8)  # (K,3)
    H, W, _ = img_hwc_u8.shape
    pix = img_hwc_u8.reshape(-1, 3).astype(np.int16)       # (N,3)
    pal = palette.astype(np.int16)                         # (K,3)

    # Compute L1 distances: (N,K)
    # Using broadcasting; N=14400, K~11 => cheap
    d = np.abs(pix[:, None, :] - pal[None, :, :]).sum(axis=-1).astype(np.int32)
    ids = d.argmin(axis=1).astype(np.uint8)

    if tol_l1 and tol_l1 > 0:
        mind = d.min(axis=1)
        ids[mind > tol_l1] = np.uint8(encoder.unknown_id)

    return ids.reshape(H, W)


def encode_ids_hybrid(img_hwc_u8: np.ndarray, encoder, tol_l1: int = 0) -> np.ndarray:
    """
    Exact match where possible, nearest fallback otherwise.
    """
    palette = np.asarray(encoder.palette, dtype=np.uint8)
    H, W, _ = img_hwc_u8.shape
    pix = img_hwc_u8.reshape(-1, 3)
    pal = palette.reshape(1, -1, 3)

    matches = (pix[:, None, :] == pal).all(axis=-1)  # (N,K)
    matched_any = matches.any(axis=1)
    ids = matches.argmax(axis=1).astype(np.uint8)

    if not matched_any.all():
        # nearest for unmatched
        pix_un = pix[~matched_any].astype(np.uint8).reshape(-1, 3)
        # reuse nearest
        d = np.abs(pix_un.astype(np.int16)[:, None, :] - palette.astype(np.int16)[None, :, :]).sum(axis=-1).astype(np.int32)
        nn = d.argmin(axis=1).astype(np.uint8)

        if tol_l1 and tol_l1 > 0:
            mind = d.min(axis=1)
            nn[mind > tol_l1] = np.uint8(encoder.unknown_id)

        ids[~matched_any] = nn

    return ids.reshape(H, W)


def summarize_ids(id_map: np.ndarray, encoder, topk: int = 8) -> str:
    flat = id_map.reshape(-1).astype(np.int32)
    c = Counter(flat.tolist())
    items = c.most_common(topk)
    parts = []
    for i, cnt in items:
        name = encoder.names[i] if 0 <= i < len(encoder.names) else f"ID{i}"
        parts.append(f"{i}:{name}({cnt})")
    return ", ".join(parts)


# -----------------------------
# Downsampling for terminal
# -----------------------------
def build_priority_from_names(encoder):
    """
    Higher priority gets preserved when downsampling.
    """
    name2id = encoder.name2id
    pr = np.zeros((len(encoder.names),), dtype=np.int32)

    def setp(name, val):
        if name in name2id:
            pr[name2id[name]] = val

    # Tune priorities for visibility
    setp("WALL", 100)
    setp("GOAL", 95)
    setp("KNIGHT", 90)
    setp("KNIGHT_ATTACK", 85)
    setp("SKELETON", 80)
    setp("BOMBKID", 80)
    setp("TURRET", 80)
    setp("ARROW", 70)
    setp("EXPLOSION", 65)
    setp("PLATFORM", 50)
    setp("BLANK", 0)

    return pr


def downsample_mode(id_map, target_h, target_w, K_guess=256):
    """
    Downsample by block-wise mode (majority vote).
    """
    H, W = id_map.shape
    if target_h >= H and target_w >= W:
        return id_map

    bh = max(1, H // target_h)
    bw = max(1, W // target_w)

    H2 = H // bh
    W2 = W // bw
    trimmed = id_map[: H2 * bh, : W2 * bw]

    out = np.zeros((H2, W2), dtype=np.uint8)
    for y in range(H2):
        for x in range(W2):
            patch = trimmed[y * bh : (y + 1) * bh, x * bw : (x + 1) * bw].ravel()
            bc = np.bincount(patch.astype(np.int32), minlength=K_guess)
            out[y, x] = int(np.argmax(bc))
    return out


def downsample_priority(id_map, target_h, target_w, priority: np.ndarray, K_guess=256):
    """
    Downsample preserving high-priority classes even if rare in the block.
    Among IDs present in a block, choose the one with maximum priority;
    break ties by count.
    """
    H, W = id_map.shape
    if target_h >= H and target_w >= W:
        return id_map

    bh = max(1, H // target_h)
    bw = max(1, W // target_w)

    H2 = H // bh
    W2 = W // bw
    trimmed = id_map[: H2 * bh, : W2 * bw]

    out = np.zeros((H2, W2), dtype=np.uint8)
    for y in range(H2):
        for x in range(W2):
            patch = trimmed[y * bh : (y + 1) * bh, x * bw : (x + 1) * bw].ravel()
            bc = np.bincount(patch.astype(np.int32), minlength=K_guess)
            present = np.flatnonzero(bc > 0)
            if present.size == 0:
                out[y, x] = 0
                continue
            # Choose by (priority, count)
            best = present[0]
            best_key = (priority[best] if best < len(priority) else 0, bc[best])
            for pid in present[1:]:
                key = (priority[pid] if pid < len(priority) else 0, bc[pid])
                if key > best_key:
                    best = pid
                    best_key = key
            out[y, x] = int(best)
    return out


# -----------------------------
# curses color helpers
# -----------------------------
def rgb_to_xterm256(r, g, b):
    """
    Approximate 24-bit RGB to xterm-256 color index.
    """
    r = int(r); g = int(g); b = int(b)

    # Grayscale
    if r == g == b:
        if r < 8:
            return 16
        if r > 248:
            return 231
        gray = int(round(((r - 8) / 247) * 24))
        return 232 + max(0, min(23, gray))

    # 6x6x6 cube
    def to_6(x):
        return int(round(x / 255 * 5))

    rr = max(0, min(5, to_6(r)))
    gg = max(0, min(5, to_6(g)))
    bb = max(0, min(5, to_6(b)))
    return 16 + 36 * rr + 6 * gg + bb


def safe_addnstr(stdscr, y, x, s, attr=0):
    maxy, maxx = stdscr.getmaxyx()
    if y < 0 or y >= maxy:
        return
    if x < 0:
        s = str(s)[-x:]
        x = 0
    if x >= maxx:
        return
    n = max(0, maxx - x - 1)
    if n <= 0:
        return
    try:
        stdscr.addnstr(y, x, str(s), n, attr)
    except curses.error:
        pass


def build_color_pairs(encoder):
    """
    Build curses color pairs for each ID in encoder palette.
    """
    curses.start_color()
    try:
        curses.use_default_colors()
    except Exception:
        pass

    palette = np.asarray(encoder.palette, dtype=np.uint8)
    K = int(palette.shape[0])

    colors_supported = getattr(curses, "COLORS", 0)
    use_256 = colors_supported >= 256

    bg_index_to_pair = {}
    id_to_pair = {}

    next_pair_id = 1
    for i in range(K):
        r, g, b = palette[i].tolist()

        if use_256:
            bg = rgb_to_xterm256(r, g, b)
        else:
            # fallback to 8 colors
            bg = curses.COLOR_BLACK
            if r > 200 and g < 80 and b < 80:
                bg = curses.COLOR_RED
            elif g > 200 and r < 80 and b < 80:
                bg = curses.COLOR_GREEN
            elif b > 200 and r < 80 and g < 80:
                bg = curses.COLOR_BLUE
            elif r > 200 and g > 200 and b < 80:
                bg = curses.COLOR_YELLOW
            elif r > 200 and b > 200 and g < 80:
                bg = curses.COLOR_MAGENTA
            elif g > 200 and b > 200 and r < 80:
                bg = curses.COLOR_CYAN
            elif r > 200 and g > 200 and b > 200:
                bg = curses.COLOR_WHITE

        if bg not in bg_index_to_pair:
            pair_id = next_pair_id
            next_pair_id += 1
            try:
                curses.init_pair(pair_id, 0, bg)
            except curses.error:
                # Pair init can fail on restricted terminals
                try:
                    curses.init_pair(pair_id, curses.COLOR_WHITE, curses.COLOR_BLACK)
                except curses.error:
                    pair_id = 0
            bg_index_to_pair[bg] = pair_id

        id_to_pair[i] = bg_index_to_pair[bg]

    return id_to_pair


def draw_map(stdscr, id_map_small, id_to_pair, top_y, left_x):
    """
    Draw id_map_small as colored blocks. Each cell uses 2 columns ("  ").
    """
    H, W = id_map_small.shape
    maxy, maxx = stdscr.getmaxyx()

    for y in range(H):
        sy = top_y + y
        if sy >= maxy:
            break
        sx = left_x
        for x in range(W):
            if sx + 1 >= maxx:
                break
            tid = int(id_map_small[y, x])
            pair = id_to_pair.get(tid, 0)
            try:
                stdscr.addstr(sy, sx, "  ", curses.color_pair(pair))
            except curses.error:
                pass
            sx += 2


def main():
    DEFAULT_GAME_PATH = "/root/immortal_suffering/immortal_suffering_linux_build.x86_64"
    DEFAULT_SEED = 42

    ap = argparse.ArgumentParser()
    ap.add_argument("--game_path", type=str, default=DEFAULT_GAME_PATH)
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--time_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=90)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--tol", type=int, default=0, help="Nearest tolerance (L1). 0 = always nearest.")
    args = ap.parse_args()

    encoder = DEFAULT_ENCODER
    palette = np.asarray(encoder.palette, dtype=np.uint8)
    K_guess = int(palette.shape[0])

    env = ImmortalSufferingEnv(
        game_path=args.game_path,
        port=args.port,
        time_scale=args.time_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
        verbose=args.verbose,
    )

    # State
    obs = env.reset()
    graphic_obs, vector_obs = parse_observation(obs)

    last_graphic_obs = graphic_obs
    last_reward = 0.0
    step_idx = 0

    # Action bits (MultiDiscrete([2]*8))
    action_bits = [0] * 8

    # UI modes
    auto_mode = True
    encoding_mode = "hybrid"   # 'exact' / 'hybrid' / 'nearest'
    channel_mode = "auto"      # 'auto' / 'rgb' / 'bgr'
    downsample_mode_kind = "mode"  # 'mode' / 'priority'

    tol_l1 = int(args.tol)

    priority = build_priority_from_names(encoder)

    def pick_channel_auto(img_u8_hwc: np.ndarray):
        """
        Decide RGB vs BGR by exact match ratio, choose better.
        """
        rgb = img_u8_hwc
        bgr = img_u8_hwc[..., ::-1]

        r_rgb = exact_match_ratio(rgb, palette)
        r_bgr = exact_match_ratio(bgr, palette)

        if r_bgr > r_rgb + 0.01:
            return "bgr", r_rgb, r_bgr
        return "rgb", r_rgb, r_bgr

    def build_id_map(img_any) -> (np.ndarray, dict):
        """
        Returns id_map, debug_info
        """
        img_u8 = to_hwc_uint8(img_any)

        debug = {}
        if channel_mode == "auto":
            chosen, r_rgb, r_bgr = pick_channel_auto(img_u8)
            debug["auto_choice"] = chosen
            debug["exact_rgb"] = r_rgb
            debug["exact_bgr"] = r_bgr
            img_u8_use = img_u8 if chosen == "rgb" else img_u8[..., ::-1]
        else:
            img_u8_use = maybe_bgr(img_u8, channel_mode)
            debug["auto_choice"] = None
            debug["exact_rgb"] = exact_match_ratio(img_u8_use if channel_mode == "rgb" else img_u8, palette)
            debug["exact_bgr"] = exact_match_ratio(img_u8_use if channel_mode == "bgr" else img_u8[..., ::-1], palette)

        debug["dtype"] = str(img_u8_use.dtype)
        debug["min"] = int(img_u8_use.min()) if img_u8_use.size else 0
        debug["max"] = int(img_u8_use.max()) if img_u8_use.size else 0

        if encoding_mode == "exact":
            id_map = encode_ids_exact(img_u8_use, encoder)
        elif encoding_mode == "nearest":
            id_map = encode_ids_nearest(img_u8_use, encoder, tol_l1=tol_l1)
        else:
            id_map = encode_ids_hybrid(img_u8_use, encoder, tol_l1=tol_l1)

        debug["top_ids"] = summarize_ids(id_map, encoder, topk=8)
        debug["shape"] = tuple(id_map.shape)
        return id_map, debug

    # initialize id_map once
    cur_id_map, cur_dbg = build_id_map(last_graphic_obs)

    def step_once(action_bits_local):
        nonlocal obs, vector_obs, last_graphic_obs, cur_id_map, cur_dbg, last_reward, step_idx

        action = np.asarray(action_bits_local, dtype=np.int64)  # (8,)
        obs, reward, done, info = env.step(action)
        done = infer_done(done, info)

        graphic_obs2, vector_obs2 = parse_observation(obs)
        last_graphic_obs = graphic_obs2
        vector_obs = vector_obs2

        cur_id_map, cur_dbg = build_id_map(last_graphic_obs)

        last_reward = float(reward)
        step_idx += 1

        if done:
            obs = env.reset()
            graphic_obs3, vector_obs3 = parse_observation(obs)
            last_graphic_obs = graphic_obs3
            vector_obs = vector_obs3
            cur_id_map, cur_dbg = build_id_map(last_graphic_obs)

    def curses_loop(stdscr):
        nonlocal auto_mode, encoding_mode, channel_mode, downsample_mode_kind, tol_l1

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(0)

        id_to_pair = build_color_pairs(encoder)
        last_tick = time.time()

        while True:
            ch = stdscr.getch()
            if ch != -1:
                if ch in (ord("q"), ord("Q")):
                    break

                if ch in (ord("a"), ord("A")):
                    auto_mode = not auto_mode

                if ch in (ord("r"), ord("R")):
                    action_bits[:] = [np.random.randint(0, 2) for _ in range(8)]

                if ch == ord("0"):
                    action_bits[:] = [0] * 8

                if ch == ord(" "):
                    step_once(action_bits)

                if ord("1") <= ch <= ord("8"):
                    i = ch - ord("1")
                    action_bits[i] = 1 - action_bits[i]

                if ch in (ord("t"), ord("T")):
                    # cycle encoding mode
                    if encoding_mode == "exact":
                        encoding_mode = "hybrid"
                    elif encoding_mode == "hybrid":
                        encoding_mode = "nearest"
                    else:
                        encoding_mode = "exact"
                    # rebuild immediately
                    nonlocal_obs_img = last_graphic_obs
                    new_map, new_dbg = build_id_map(nonlocal_obs_img)
                    globals()["cur_id_map"] = new_map
                    globals()["cur_dbg"] = new_dbg

                if ch in (ord("c"), ord("C")):
                    # cycle channel mode
                    if channel_mode == "auto":
                        channel_mode = "rgb"
                    elif channel_mode == "rgb":
                        channel_mode = "bgr"
                    else:
                        channel_mode = "auto"
                    new_map, new_dbg = build_id_map(last_graphic_obs)
                    globals()["cur_id_map"] = new_map
                    globals()["cur_dbg"] = new_dbg

                if ch in (ord("m"), ord("M")):
                    downsample_mode_kind = "priority" if downsample_mode_kind == "mode" else "mode"

                if ch == ord("["):
                    tol_l1 = max(0, tol_l1 - 2)
                    new_map, new_dbg = build_id_map(last_graphic_obs)
                    globals()["cur_id_map"] = new_map
                    globals()["cur_dbg"] = new_dbg

                if ch == ord("]"):
                    tol_l1 = min(200, tol_l1 + 2)
                    new_map, new_dbg = build_id_map(last_graphic_obs)
                    globals()["cur_id_map"] = new_map
                    globals()["cur_dbg"] = new_dbg

            now = time.time()
            if auto_mode and (now - last_tick) >= (1.0 / max(1e-6, float(args.fps))):
                step_once(action_bits)
                last_tick = now

            stdscr.erase()
            maxy, maxx = stdscr.getmaxyx()

            # Header lines
            safe_addnstr(
                stdscr, 0, 0,
                f"Step={step_idx} Auto={auto_mode} FPS={args.fps:.1f} "
                f"Action={''.join(map(str, action_bits))}  "
                f"[t:enc({encoding_mode}) c:ch({channel_mode}) m:down({downsample_mode_kind}) "
                f"tol={tol_l1}] "
                f"(1-8 toggle, space step, a auto, r random, 0 clear, q quit, [ ] tol)"
            )
            safe_addnstr(stdscr, 1, 0, f"last_reward={last_reward:.3f}  id_map_shape={cur_dbg.get('shape')}  img_u8(min,max)=({cur_dbg.get('min')},{cur_dbg.get('max')})")

            # Exact ratio debug
            if channel_mode == "auto":
                safe_addnstr(
                    stdscr, 2, 0,
                    f"auto_choice={cur_dbg.get('auto_choice')}  exact_rgb={cur_dbg.get('exact_rgb',0)*100:.1f}%  exact_bgr={cur_dbg.get('exact_bgr',0)*100:.1f}%  term={maxy}x{maxx}"
                )
            else:
                # show both ratios anyway
                safe_addnstr(
                    stdscr, 2, 0,
                    f"exact_rgb={cur_dbg.get('exact_rgb',0)*100:.1f}%  exact_bgr={cur_dbg.get('exact_bgr',0)*100:.1f}%  term={maxy}x{maxx}"
                )

            # Vector info
            if vector_obs is not None and len(vector_obs) > IDX_TIME:
                dmg = float(vector_obs[IDX_CUM_DAMAGE])
                actionable = float(vector_obs[IDX_IS_ACTIONABLE])
                gdist = float(vector_obs[IDX_GOAL_DIST])
                tval = float(vector_obs[IDX_TIME])
                safe_addnstr(stdscr, 3, 0, f"vec: damage={dmg:.2f} actionable={actionable:.0f} goal_dist={gdist:.2f} time={tval:.2f}")

            safe_addnstr(stdscr, 4, 0, f"top_ids: {cur_dbg.get('top_ids','')}")

            # Map area
            top = 6
            avail_rows = max(1, maxy - top - 1)
            avail_cols = max(1, (maxx - 2) // 2)

            if cur_id_map.ndim != 2:
                safe_addnstr(stdscr, top, 0, f"[WARN] id_map has shape {cur_id_map.shape}, expected (H,W)")
            else:
                if downsample_mode_kind == "priority":
                    id_map_small = downsample_priority(cur_id_map, target_h=avail_rows, target_w=avail_cols, priority=priority, K_guess=K_guess)
                else:
                    id_map_small = downsample_mode(cur_id_map, target_h=avail_rows, target_w=avail_cols, K_guess=K_guess)

                draw_map(stdscr, id_map_small, id_to_pair, top_y=top, left_x=0)

            stdscr.refresh()
            time.sleep(0.01)

    try:
        curses.wrapper(curses_loop)
    finally:
        env.close()


if __name__ == "__main__":
    main()
