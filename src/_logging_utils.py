"""Internal pipeline progress reporting and run configuration utilities."""
from __future__ import annotations

import random as _random
import time as _time
import unicodedata as _unicodedata
from pathlib import Path


_TOTAL_STEPS = 24
_MENU_PAUSE = 3.0         # seconds before hype text appears after a menu choice
_CONFIG_HYPE_PAUSE = 7.0  # seconds after hype text before the next config step appears
_STEP_FLOOR = 5.0         # minimum seconds each pipeline step stays visible
_STARTUP_TIME = 12.0      # seconds to display the title banner before the pun lines reveal
_BOOT_PAUSE = 4.0         # seconds of "loading" silence after pressing Enter
_STEP_EMOJI = [
    "🐱", "🌸", "🌷", "🌼", "🌻", "🌹", "🌺", "💖", "✨", "🦄",
    "🎀", "💕", "🍓", "🧁", "🌈", "🦋", "🪻", "🌙", "⭐", "🐰",
    "💗", "🍡", "🪐",
]

_TRAIL_EMOJI = [
    "🌷✨", "💕🎀", "🌸💖", "🦋✨", "🌟💗", "🍓💞", "🧁💕", "🪻💖",
    "🌼✨", "🐱💕", "🦄🌈", "🌹💖", "💝🌸", "🌺✨", "🎀💗",
]

_KAOMOJI = [
    "(◕‿◕✿)", "( ˘͈ ᵕ ˘͈♡)", "(✿◠‿◠)", "(｡♥‿♥｡)", "ヾ(＾∇＾)",
    "(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧", "(✿ ♡‿♡)", "(*˘︶˘*).｡.:*♡", "٩(◕‿◕)۶",
    "(づ｡◕‿‿◕｡)づ", "ʕ•́ᴥ•̀ʔっ♡", "(ᵔᴥᵔ)", "(っ◔◡◔)っ ♥", "(♡˙︶˙♡)",
    "ヾ(＾-＾)ノ", "(˶ᵔ ᵕ ᵔ˶)♡", "ʚ♡⃛ɞ", "(っ˘ω˘ς )", "(/^▽^)/",
    "(✯◡✯)", "(◍•ᴗ•◍)❤", "(*✧×✧*)",
]

_VERBS = [
    "✨ done in", "🌸 finished in", "💕 wrapped up in", "🎀 all done in",
    "💖 completed in", "🌷 ready in", "✨ baked in", "💗 prepped in",
]

_RNG = _random.Random(42)

_STATE = {"step": 0, "t0": 0.0}


def _step_start(label: str) -> float:
    _STATE["step"] += 1
    n = _STATE["step"]
    emoji = _STEP_EMOJI[(n - 1) % len(_STEP_EMOJI)]
    trail = _TRAIL_EMOJI[(n - 1) % len(_TRAIL_EMOJI)]

    content = f"  │ [{n:2d}/{_TOTAL_STEPS}] {emoji} {label}… {trail}"
    interior_w = _vw(content) - 4

    seg = "─ ✿ ─"
    seg_w = _vw(seg)

    top = f"─ {emoji} ─" + seg + f"─ {trail} ─"
    tw = _vw(top)
    while tw + seg_w <= interior_w:
        top += seg
        tw += seg_w
    while tw < interior_w:
        top += "─"
        tw += 1

    bot = ""
    bw = 0
    while bw + seg_w <= interior_w:
        bot += seg
        bw += seg_w
    while bw < interior_w:
        bot += "─"
        bw += 1

    print(f"\n  ╭{top}╮")
    print(f"  │ [{n:2d}/{_TOTAL_STEPS}] {emoji} {label}… {trail}")
    print(f"  ╰{bot}╯", flush=True)
    return _time.time()


def _step_end(t_start: float, message: str) -> None:
    elapsed = _time.time() - t_start
    kao = _KAOMOJI[_STATE["step"] % len(_KAOMOJI)]
    verb = _VERBS[_STATE["step"] % len(_VERBS)]
    print(f"        🌷 {message}")
    print(f"        💕 {kao}  {verb} {elapsed:5.1f}s  ✧･ﾟ:*", flush=True)
    remaining = _STEP_FLOOR - elapsed
    if remaining > 0:
        _time.sleep(remaining)


def _banner_start() -> None:
    _STATE["step"] = 0
    _STATE["t0"] = _time.time()
    print("")
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    print("                                                                   ")
    print(f"        🦄✨   starting up the kawaii train   🚂🎀✨               ")
    print(f"        🌸💕   {_TOTAL_STEPS} cute little steps to chug through   💕🌸  ")
    print(f"        🫎♡    please be patient, lots of moose to count    ♡🫎   ")
    print(f"        🌷    ٩(◕‿◕✿)۶  ✧･ﾟ:*  here we go!!  *:･ﾟ✧  (◕‿◕✿)۶    🌷")
    print("                                                                   ")
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ", flush=True)


def _banner_end(trees: list[dict] | Path, *legacy_args: Path) -> None:
    """Print the closing kawaii banner with one path-block per fold strategy.

    New API: ``_banner_end([{label, output_dir, models_dir, figures_dir,
    species_output_dir}, ...])``. Legacy API ``_banner_end(output_dir,
    models_dir, figures_dir)`` is kept for any caller that still uses it.
    """
    if isinstance(trees, Path):
        trees = [{
            "label":              "month",
            "output_dir":         trees,
            "models_dir":         legacy_args[0],
            "figures_dir":        legacy_args[1],
            "species_output_dir": None,
        }]

    total = _time.time() - _STATE["t0"]
    mins, secs = divmod(int(total), 60)
    print("")
    print("  🌈✨💖🎀🌸💕  ✧･ﾟ:*✧･ﾟ:*  P I P E L I N E   C O M P L E T E !!  *:･ﾟ✧*:･ﾟ✧  💕🌸🎀💖✨🌈")
    print("                                                                                          ")
    print("        ⋆｡˚ ⋆｡˚ ⋆｡˚    ٩( ๑•̀o•́๑ )و    ⋆｡˚ ⋆｡˚ ⋆｡˚                                       ")
    print(f"             🐱   total time     ➜   {mins:>2}m {secs:>2}s   ( ´ ▽ ` )ﾉ ♡                ")
    print(f"             🌸   steps done     ➜   {_STATE['step']}/{_TOTAL_STEPS}  ✓✓✓ ✧･ﾟ:*           ")
    for tree in trees:
        label = tree.get("label", "?")
        fu_emoji = "📆" if label == "month" else "🗓️" if label == "year" else "✨"
        fu_label = {"month": "monthly folds", "year": "yearly folds"}.get(label, label)
        print(f"             {fu_emoji}   {fu_label}                                                  ")
        print(f"               🌷   CSVs       ➜   {tree['output_dir']}                                ")
        print(f"               🦄   models     ➜   {tree['models_dir']}                                ")
        print(f"               🎀   figures    ➜   {tree['figures_dir']}                               ")
        if tree.get("species_output_dir") is not None:
            print(f"               🦌   per-species ➜   {tree['species_output_dir']}                   ")
    print("                                                                                          ")
    print("        ✿  💕  ✿  💖  ✿  💕  ✿  💖  ✿  💕  ✿  💖  ✿  💕  ✿  💖  ✿                   ")
    print("                                                                                          ")
    print("        🌸💖💕   thank you for running, Amanda!!   💕💖🌸                                 ")
    print("        🫎✨💕   you did an absoMOOSEly amazing job!!   💕✨🫎                            ")
    print("        🦌🎀🌸   tROEdeer-ly impressive results!! the deer are proud!!   🌸🎀🦌           ")
    print("        🐗💖✨   wild-BOARsome analysis!! absolutely feral science!!   ✨💖🐗             ")
    print("        🌷💕🦄   now go celebrate — you've earned it!!   🦄💕🌷                          ")
    print("        🌸🎀✨   and don't forget to have an awesome, wonderful, sparkly day!!!!   ✨🎀🌸      ")
    print("                                                                                          ")
    print("              ╭─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─╮                                              ")
    print("              │   ⋆｡˚    ٩(◕‿◕✿)۶    ˚｡⋆     │                                          ")
    print("              ╰─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─♡─╯                                              ")
    print("                                                                                          ")
    print("  🌸✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡✿🌸  ")
    print("")


def _kawaii_pause(seconds: float) -> None:
    _time.sleep(seconds)


def _vw(s: str) -> int:
    """Visual column width: Wide/Fullwidth chars count as 2, everything else as 1."""
    return sum(2 if _unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _kawaii_section_banner(title: str, emoji: str = "🌸") -> None:
    interior_w = _vw(f"  ✨  {title}  ✨") + 1  # +1 for the │ becoming part of alignment
    seg = "─ ✿ ─"
    eseg = f"─ {emoji} ─"
    seg_w = _vw(seg)
    eseg_w = _vw(eseg)

    top = eseg + seg + eseg
    tw = eseg_w + seg_w + eseg_w
    while tw + seg_w <= interior_w:
        top += seg
        tw += seg_w
    while tw < interior_w:
        top += "─"
        tw += 1

    bot = ""
    bw = 0
    while bw + seg_w <= interior_w:
        bot += seg
        bw += seg_w
    while bw < interior_w:
        bot += "─"
        bw += 1

    print(f"\n  ╭{top}╮")
    print(f"  │  ✨  {title}  ✨")
    print(f"  ╰{bot}╯\n")


def _interactive_menu() -> tuple[str | None, str, str, str]:
    """Show run-configuration menu; return (species_filter, species_mode, species_variant, fold_unit)."""
    print("\n  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    print("                                                                   ")
    print("       🌸💖 ✧ W I L D L I F E   C O L L I S I O N ✧ 💖🌸          ")
    print("       🌷✨ ✧     P R E D I C T I O N   ⋆ M O D E L     ✧ ✨🌷    ")
    print("                                                                   ")
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    print()
    print("        💕🐱  hello Amanda!! so happy you are here today!!  🐱💕  ")
    print("        🌷✨  the animals are waiting and they are SO excited  ✨🌷")
    print("        🌸✨  Firing up the program, please admire the banner!!  ✨🌸", flush=True)
    print()
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    _kawaii_pause(_STARTUP_TIME)

    print()
    print("        🫎💕   the moose have finished loading and they are MOOSE-T excited to see you!!")
    print("        🦌✨   ROE-dy and waiting, just for you!!               ")
    print("        🐗🌸   this is going to be BOAR-illiant science!!        ")
    print("        🌷💖   absoFAWN-lutely cannot wait to get started!!      ")
    print()
    print("  ╔♡══════════════════════════════════════════════════════════♡╗")
    print("  ║                                                            ║")
    print("  ║      ✨ 💖   P R E S S   E N T E R   💖 ✨                 ║")
    print("  ║         to board the kawaii train!!                        ║")
    print("  ║         🚂💨  ALL ABOARD!!  💨🚂                           ║")
    print("  ║                                                            ║")
    print("  ╚♡══════════════════════════════════════════════════════════♡╝")
    print()
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    input("")

    print("        🌈💕  OH WOW LET'S GO!! the moose have been waiting ALL DAY for this!!  💕🌈")
    print("        🦄✨  you are literally their favourite scientist!! they are SO happy!!  ✨🦄")
    _kawaii_pause(_STEP_FLOOR)
    print()
    print("        🚂💨  hang on tight — firing up the program!!  💨🚂      ")
    print("        🌸✨  loading all the moose data, please stand by!!  ✨🌸", flush=True)
    print()
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    _kawaii_pause(_BOOT_PAUSE)
    _kawaii_pause(_STEP_FLOOR)

    _silly = "  (ಠ_ಠ)  that's not an option, silly!! choose again!!  (ಠ_ಠ)"

    # ── Step 0 (asked first, applies to whatever else gets chosen) ──
    _kawaii_section_banner(
        "Amanda, we know you absolutely HATE folding the laundry... "
        "but at least now you can fold the DATA!! 💕💖💗 (｡♥‿♥｡)",
        "🧺",
    )
    print("  ✨💖  the ONLY kind of folding you'll actually enjoy!! pinky promise!! 🤞🌸")
    print("  🌷✨  (no socks were harmed in the making of this cross-validation!!)  ヾ(＾∇＾)ﾉ ♡")
    print()
    print("  [1] 📆💕 Monthly folds (default)  ✿ ٩(◕‿◕✿)۶ ✿")
    print("       🌸  the classic crease!! every little month gets its own teeny tiny fold!!  🌸")
    print("       💖  expanding window  •  1 moose-nth test horizon  •  ~120 dainty folds  💖")
    print("       ✨  (the OG setup!! every metric in the thesis so far comes from these!!)  ✨")
    print()
    print("  [2] 🗓️✨ Yearly folds  ✿ (｡>‿‿<｡)♡ ✿")
    print("       🌷  the chunky burrito fold!! one whole year-iversary per test window!!  🌷")
    print("       💗  expanding window  •  1 year test horizon  •  ~10 thicc folds  💗")
    print("       🦄  (Henki's request!! fewer folds, each test window is a full calendar year!!)  🦄")
    print()
    print("  [3] 🌈💖 BOTH (the laundry-day special!!)  ✿ (づ｡◕‿‿◕｡)づ♡♡♡ ✿")
    print("       ✨  fold the data BY MOON-TH and BY YEAR!! TimeFolding™!! double-cute!!  ✨")
    print("       🌸  monthly results land in outputs/  •  yearly results land in outputs_year/  🌸")
    print("       💕  (it's like sorting whites from colours, but for TIME itself!! very tidy!!)  💕")
    print("       🎀  warning: maximum thesis-origami!! prepare to be folded WITH joy!!  🎀\n")
    while True:
        fu_choice = input("  Choice [1]: ").strip() or "1"
        if fu_choice in ("1", "2", "3"):
            break
        print(_silly)
    fold_unit = {"1": "month", "2": "year", "3": "both"}[fu_choice]
    fold_unit_label = {
        "month": "monthly folds 📆 (the classic crease!!) 💕",
        "year":  "yearly folds 🗓️ (the burrito fold!!) ✨  — Henki mode!! 🦄",
        "both":  "BOTH 📆🗓️ (LAUNDRY DAY SPECIAL!!) 💖💖  — outputs/ + outputs_year/",
    }[fold_unit]
    fold_unit_hype = {
        "month": "monthly folds!! the classic crease!! every metric we have was folded this way!! 📆💕  (◕‿◕✿)",
        "year":  "yearly folds!! Henki vibes!! one whole calendar year per fold — chunky-chunky-chunky!! 🗓️✨  ٩(◕‿◕)۶",
        "both":  "BOTH!!! we are folding time TWICE!! moose-nthly AND year-iversary style!! ultimate origami!! 🌟🌟  (｡♥‿♥｡)",
    }[fold_unit]
    _kawaii_pause(_MENU_PAUSE)
    print(f"\n  💕✨  {fold_unit_hype}")
    print(f"  🌸  {fold_unit_label} locked in!! 🌸  ✧･ﾟ:* (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    _kawaii_pause(_CONFIG_HYPE_PAUSE)

    while True:
        _kawaii_section_banner("what would you like to do today, Amanda?? ✿(◕‿◕✿)", "🦄")
        print("  [1] 🌈 Full sweep  — pooled + ALL per-species   (~55 min)  ← recommended!! 💖")
        print("  [2] 🐱 Pooled only — original baseline          (~4 min)   ← quick & classic")
        print("  [3] 🎀 Configure   — i choose my own adventure!!\n")

        while True:
            choice = input("  Choice [1]: ").strip() or "1"
            if choice in ("1", "2", "3"):
                break
            print(_silly)

        if choice == "1":
            _kawaii_pause(_MENU_PAUSE)
            print()
            print("  ✨💖✨  OH WOW BEST CHOICE AMANDA!! the FULL SWEEP!! ✨💖✨")
            print("  🌸  all four species!! all modes!! all variants!! every single combo!!  🌸")
            print("  🦄  24 beautiful per-species models PLUS the pooled baseline!!  🦄")
            print("  💕  this is going to be SPECTACULAR and we are SO proud of you  💕")
            _kawaii_pause(_MENU_PAUSE)
            return "all", "all", "both", fold_unit

        if choice == "2":
            _kawaii_pause(_MENU_PAUSE)
            print()
            print("  🐱💕  a classic!! the original pooled pipeline!! timeless!! elegant!!  💕🐱")
            print("  🌷  all four species together as ONE beautiful unified model  🌷")
            print("  ✨  clean, fast, powerful — peak wildlife science!!  ✨")
            _kawaii_pause(_MENU_PAUSE)
            return None, "default", "lag", fold_unit

        _kawaii_pause(_MENU_PAUSE)
        print()
        print("  🎀💕  ooh a custom adventure!! let us build your perfect run together!!  💕🎀")
        print("  🌸  answer three tiny questions and we will make magic happen!!  🌸")
        _kawaii_pause(_MENU_PAUSE)

        _kawaii_section_banner("step 1 of 3 — which species?? 🦌🐗", "🌿")
        print("  [1] 🌈 All species   — everyone deserves love!!")
        print("  [2] 🦌 Roe deer      — elegant and speedy!!")
        print("  [3] 🫎 Moose         — the big majestic one!!")
        print("  [4] 🐗 Wild boar     — chaotic and wonderful!!")
        print("  [5] 🦌 Fallow deer   — fancy and fabulous!!\n")
        while True:
            sp = input("  Choice [1]: ").strip() or "1"
            if sp in ("1", "2", "3", "4", "5"):
                break
            print(_silly)
        species = {"1": "all", "2": "roe_deer", "3": "moose", "4": "wild_boar", "5": "fallow_deer"}[sp]
        sp_label = {"all": "ALL SPECIES 🌈", "roe_deer": "roe deer 🦌", "moose": "moose 🫎",
                    "wild_boar": "wild boar 🐗", "fallow_deer": "fallow deer 🦌"}[species]
        sp_hype = {
            "all":         "ALL SPECIES!! every single one!! the full squad!! nobody left behind!! 🌈💖",
            "roe_deer":    "roe deer!! so graceful!! so fast!! such tiny hooves!! 🦌✨",
            "moose":       "MOOSE!! the icon!! the legend!! the big beautiful baby!! 🫎💕",
            "wild_boar":   "wild boar!! chaotic energy!! absolute unit!! love the commitment!! 🐗🔥",
            "fallow_deer": "fallow deer!! fancy spots!! very distinguished taste!! 🦌👑",
        }[species]
        _kawaii_pause(_MENU_PAUSE)
        print(f"\n  💕✨  {sp_hype}")
        print(f"  🌸  {sp_label} locked in!! perfect choice!!  🌸")
        _kawaii_pause(_CONFIG_HYPE_PAUSE)

        _kawaii_section_banner("step 2 of 3 — which infrastructure mode?? 🚂🛣️", "🌺")
        print("  [1] 🌍 All collisions   — road + rail together, the whole picture!!")
        print("  [2] 🛣️  Road only        — classic road WVC analysis!!")
        print("  [3] 🚂 Rail only        — brave!! very niche!! we love the audacity!!")
        print("  [4] 🔀 Both separately  — road AND rail, two analyses in one!!\n")
        while True:
            md = input("  Choice [1]: ").strip() or "1"
            if md in ("1", "2", "3", "4"):
                break
            print(_silly)
        mode = {"1": "default", "2": "road", "3": "rail", "4": "both"}[md]
        mode_label = {"default": "ALL COLLISIONS 🌍", "road": "road only 🛣️",
                      "rail": "rail only 🚂", "both": "road + rail separately 🔀"}[mode]
        mode_hype = {
            "default": "all collisions!! the full picture!! road AND rail together in beautiful harmony!! 🌍💕",
            "road":    "road only!! the classic!! where it all started!! roads roads roads!! 🛣️✨",
            "rail":    "RAIL ONLY!! so brave!! so specific!! the trains will not be ignored!! 🚂💖",
            "both":    "BOTH modes separately!! twice the analysis!! twice the science!! double the moose!! 🔀🦄",
        }[mode]
        _kawaii_pause(_MENU_PAUSE)
        print(f"\n  💕✨  {mode_hype}")
        print(f"  🌸  {mode_label} locked in!! outstanding decision!!  🌸")
        _kawaii_pause(_CONFIG_HYPE_PAUSE)

        _kawaii_section_banner("step 3 of 3 — lag or no-lag?? 🔮📊", "💫")
        print("  [1] 🔮 Lag      — forecast model   (uses last month's collisions as a hint!)")
        print("  [2] 📊 No-lag   — determinants     (pure environmental features only!! very scientific!!)")
        print("  [3] ✨ Both     — run both variants — maximum science!!\n")
        while True:
            vr = input("  Choice [1]: ").strip() or "1"
            if vr in ("1", "2", "3"):
                break
            print(_silly)
        variant = {"1": "lag", "2": "no-lag", "3": "both"}[vr]
        var_label = {"lag": "lag (forecast) 🔮", "no-lag": "no-lag (determinants) 📊", "both": "BOTH variants ✨"}[variant]
        var_hype = {
            "lag":    "lag features!! using the past to predict the future!! very time-series-pilled!! 🔮💕",
            "no-lag": "no-lag!! pure environmental determinants!! what CAUSES the collisions?? very deep!! 📊🌿",
            "both":   "BOTH variants!! the full comparison!! this IS the thesis contribution!! 🏆✨💖",
        }[variant]
        _kawaii_pause(_MENU_PAUSE)
        print(f"\n  💕✨  {var_hype}")
        print(f"  🌸  {var_label} locked in!! you are NAILING this!!  🌸")
        _kawaii_pause(_CONFIG_HYPE_PAUSE)

        print()
        print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
        print(f"       🎀  CONFIGURATION COMPLETE!!  🎀")
        print(f"       💖  fold     : {fold_unit_label}")
        print(f"       💖  species  : {sp_label}")
        print(f"       💖  mode     : {mode_label}")
        print(f"       💖  variant  : {var_label}")
        print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
        print()
        print("  [y] ✅  yes!! lock it in and let's GO!!  💖")
        print("  [n] 🔄  hmm actually... let me choose again!!\n")
        while True:
            confirm = input("  Confirm [Y/n]: ").strip().lower() or "y"
            if confirm in ("y", "n"):
                break
            print(_silly)

        if confirm == "y":
            _kawaii_pause(_MENU_PAUSE)
            return species, mode, variant, fold_unit

        print()
        print("  💕  no worries!! let's start over!! the moose are very patient!!  💕")
        _kawaii_pause(_MENU_PAUSE)


def _launch_fanfare() -> None:
    print()
    print("  🚂💨💨  READY TO FIRE UP THE KAWAII TRAIN?!  💨💨🚂")
    print("  🌸  CHUG CHUG CHUG CHUG CHUG CHUG CHUG CHUG!!  🌸")
    print("  💖  here we GOOOOO Amanda hold on tight!!  💖")
    print()
    _kawaii_pause(_MENU_PAUSE)
