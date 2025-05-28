# ====================  concurrency_fix.py  ====================
"""
Real-time GPU-concurrency fix for transcript_generator()
"""
import asyncio
import inspect
import importlib
import os

# ------------------------------------------------------------------
# 1.  Locate the original transcript_generator (wherever you import it)
# ------------------------------------------------------------------
ORIGINAL_MODULE_PATH = "WTranscriptor.utils.utils"   # adjust only if path changes
ORIGINAL_FUNC_NAME   = "transcript_generator"        # keep identical

orig_mod  = importlib.import_module(ORIGINAL_MODULE_PATH)
orig_func = getattr(orig_mod, ORIGINAL_FUNC_NAME)

if not inspect.iscoroutinefunction(orig_func):
    raise RuntimeError(
        f"{ORIGINAL_MODULE_PATH}.{ORIGINAL_FUNC_NAME} "
        "must be an async function for the semaphore wrapper to work."
    )

# ------------------------------------------------------------------
# 2.  Create ONE shared semaphore
# ------------------------------------------------------------------
MAX_GPU_CONCURRENCY = int(os.getenv("GPU_MAX_CONCURRENCY", 8))   # default 8
_GPU_SEMAPHORE = asyncio.Semaphore(MAX_GPU_CONCURRENCY)

# ------------------------------------------------------------------
# 3.  Define the wrapper
# ------------------------------------------------------------------
async def _transcript_generator_with_limits(*args, **kwargs):
    """
    Proxy that waits for a free GPU slot (usually a few ms) and
    then delegates to the original implementation.
    """
    async with _GPU_SEMAPHORE:        # <- ** the single throttle point **
        return await orig_func(*args, **kwargs)

# ------------------------------------------------------------------
# 4.  Monkey-patch the symbol so everybody gets the limited version
# ------------------------------------------------------------------
setattr(orig_mod, ORIGINAL_FUNC_NAME, _transcript_generator_with_limits)

# ------------------------------------------------------------------
# 5.  (optional) expose the semaphore for metrics / health checks
# ------------------------------------------------------------------
def gpu_slots_in_use() -> int:
    return MAX_GPU_CONCURRENCY - _GPU_SEMAPHORE._value
