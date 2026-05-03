"""Internal pipeline progress reporting and run configuration utilities."""
from __future__ import annotations

import random as _random
import time as _time
from pathlib import Path


_TOTAL_STEPS = 24
_MENU_PAUSE = 2.0   # seconds between menu interactions — edit this to speed up or slow down
_STEP_FLOOR = 2.0   # minimum seconds each pipeline step stays visible

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
    print(f"\n  ╭─ ✿ ─ {emoji} ─ ✿ ─ ✿ ─ {trail} ─ ✿ ─ ✿ ─ ✿ ─╮")
    print(f"  │ [{n:2d}/{_TOTAL_STEPS}] {emoji} {label}… {trail}")
    print(f"  ╰─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─╯", flush=True)
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


def _banner_end(output_dir: Path, models_dir: Path, figures_dir: Path) -> None:
    total = _time.time() - _STATE["t0"]
    mins, secs = divmod(int(total), 60)
    print("")
    print("  🌈✨💖🎀🌸💕  ✧･ﾟ:*✧･ﾟ:*  P I P E L I N E   C O M P L E T E !!  *:･ﾟ✧*:･ﾟ✧  💕🌸🎀💖✨🌈")
    print("                                                                                          ")
    print("        ⋆｡˚ ⋆｡˚ ⋆｡˚    ٩( ๑•̀o•́๑ )و    ⋆｡˚ ⋆｡˚ ⋆｡˚                                       ")
    print(f"             🐱   total time     ➜   {mins:>2}m {secs:>2}s   ( ´ ▽ ` )ﾉ ♡                ")
    print(f"             🌸   steps done     ➜   {_STATE['step']}/{_TOTAL_STEPS}  ✓✓✓ ✧･ﾟ:*           ")
    print(f"             🌷   CSVs           ➜   {output_dir}                                          ")
    print(f"             🦄   models         ➜   {models_dir}                                          ")
    print(f"             🎀   figures        ➜   {figures_dir}                                         ")
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


def _kawaii_pause(seconds: float = 2.0) -> None:
    _time.sleep(seconds)


def _kawaii_section_banner(title: str, emoji: str = "🌸") -> None:
    print(f"\n  ╭─ ✿ ─ {emoji} ─ ✿ ─ ✿ ─ {emoji} ─ ✿ ─ ✿ ─ ✿ ─╮")
    print(f"  │  ✨  {title}  ✨")
    print(f"  ╰─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─ ✿ ─╯\n")


def _interactive_menu() -> tuple[str | None, str, str]:
    """Show run-configuration menu; return (species_filter, species_mode, species_variant)."""
    print("\n  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    print("                                                                   ")
    print("       🌸💖 ✧ W I L D L I F E   C O L L I S I O N ✧ 💖🌸          ")
    print("       🌷✨ ✧     P R E D I C T I O N   ⋆ M O D E L     ✧ ✨🌷    ")
    print("                                                                   ")
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    print()
    print("        💕🐱  hello Amanda!! so happy you are here today!!  🐱💕  ")
    print("        🌷✨  the moose are waiting and they are SO excited  ✨🌷  ")
    print()
    print("        🚂💨  hang on tight — firing up the program!!  💨🚂      ")
    print("        🌸✨  loading all the moose data, please admire the banner  ✨🌸")
    print()
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ", flush=True)
    print()
    _kawaii_pause(10.0)

    _kawaii_section_banner("what shall we run today?? ✿(◕‿◕✿)", "🦄")
    print("  [1] 🌈 Full sweep  — pooled + ALL per-species   (~55 min)  ← recommended!! 💖")
    print("  [2] 🐱 Pooled only — original baseline          (~4 min)   ← quick & classic")
    print("  [3] 🎀 Configure   — i choose my own adventure!!\n")

    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "1":
        _kawaii_pause(_MENU_PAUSE)
        print()
        print("  ✨💖✨  OH WOW BEST CHOICE AMANDA!! the FULL SWEEP!! ✨💖✨")
        print("  🌸  all four species!! all modes!! all variants!! every single combo!!  🌸")
        print("  🦄  24 beautiful per-species models PLUS the pooled baseline!!  🦄")
        print("  💕  this is going to be SPECTACULAR and we are SO proud of you  💕")
        _kawaii_pause(_MENU_PAUSE)
        return "all", "all", "both"

    if choice == "2":
        _kawaii_pause(_MENU_PAUSE)
        print()
        print("  🐱💕  a classic!! the original pooled pipeline!! timeless!! elegant!!  💕🐱")
        print("  🌷  all four species together as ONE beautiful unified model  🌷")
        print("  ✨  clean, fast, powerful — peak wildlife science!!  ✨")
        _kawaii_pause(_MENU_PAUSE)
        return None, "default", "lag"

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
    sp = input("  Choice [1]: ").strip() or "1"
    species = {"1": "all", "2": "roe_deer", "3": "moose", "4": "wild_boar", "5": "fallow_deer"}.get(sp, "all")
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
    _kawaii_pause(_MENU_PAUSE)

    _kawaii_section_banner("step 2 of 3 — which infrastructure mode?? 🚂🛣️", "🌺")
    print("  [1] 🌍 All collisions   — road + rail together, the whole picture!!")
    print("  [2] 🛣️  Road only        — classic road WVC analysis!!")
    print("  [3] 🚂 Rail only        — brave!! very niche!! we love the audacity!!")
    print("  [4] 🔀 Both separately  — road AND rail, two analyses in one!!\n")
    md = input("  Choice [1]: ").strip() or "1"
    mode = {"1": "default", "2": "road", "3": "rail", "4": "both"}.get(md, "default")
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
    _kawaii_pause(_MENU_PAUSE)

    _kawaii_section_banner("step 3 of 3 — lag or no-lag?? 🔮📊", "💫")
    print("  [1] 🔮 Lag      — forecast model   (uses last month's collisions as a hint!)")
    print("  [2] 📊 No-lag   — determinants     (pure environmental features only!! very scientific!!)")
    print("  [3] ✨ Both     — run both variants — maximum science!!\n")
    vr = input("  Choice [1]: ").strip() or "1"
    variant = {"1": "lag", "2": "no-lag", "3": "both"}.get(vr, "lag")
    var_label = {"lag": "lag (forecast) 🔮", "no-lag": "no-lag (determinants) 📊", "both": "BOTH variants ✨"}[variant]
    var_hype = {
        "lag":    "lag features!! using the past to predict the future!! very time-series-pilled!! 🔮💕",
        "no-lag": "no-lag!! pure environmental determinants!! what CAUSES the collisions?? very deep!! 📊🌿",
        "both":   "BOTH variants!! the full comparison!! this IS the thesis contribution!! 🏆✨💖",
    }[variant]
    _kawaii_pause(_MENU_PAUSE)
    print(f"\n  💕✨  {var_hype}")
    print(f"  🌸  {var_label} locked in!! you are NAILING this!!  🌸")
    _kawaii_pause(_MENU_PAUSE)

    print()
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    print(f"       🎀  CONFIGURATION COMPLETE!!  🎀")
    print(f"       💖  species  : {sp_label}")
    print(f"       💖  mode     : {mode_label}")
    print(f"       💖  variant  : {var_label}")
    print("  ✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡.:* *.:｡✿*ﾟ'ﾟ･✿.｡  ")
    _kawaii_pause(_MENU_PAUSE)
    return species, mode, variant


def _launch_fanfare() -> None:
    print()
    print("  🚂💨💨  READY TO FIRE UP THE KAWAII TRAIN?!  💨💨🚂")
    print("  🌸  CHUG CHUG CHUG CHUG CHUG CHUG CHUG CHUG!!  🌸")
    print("  💖  here we GOOOOO Amanda hold on tight!!  💖")
    print()
    _kawaii_pause(_MENU_PAUSE)
