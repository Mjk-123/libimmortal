try:
    from mlagents_envs.base_env import TerminalSteps
except Exception:
    TerminalSteps = None

try:
    # Raised when calling step() after done=True without reset()
    from mlagents_envs.envs.unity_gym_env import UnityGymException
except Exception:
    UnityGymException = None

try:
    from mlagents_envs.exception import UnityTimeOutException
except Exception:
    UnityTimeOutException = None

try:
    from mlagents_envs.exception import UnityEnvironmentException
except Exception:
    UnityEnvironmentException = None

from libimmortal.env import ImmortalSufferingEnv

from libimmortal.samples.PPO.utils.ddp import is_main_process


def _is_timeout_exc(e: Exception) -> bool:
    return (UnityTimeOutException is not None) and isinstance(e, UnityTimeOutException)

def _is_unity_env_unloaded_exc(e: Exception) -> bool:
    if UnityEnvironmentException is None:
        return False
    if not isinstance(e, UnityEnvironmentException):
        return False
    msg = str(e)
    # Be conservative: restart only for common transient/comm/unloaded cases
    if "No Unity environment is loaded" in msg:
        return True
    if "timed out" in msg.lower():
        return True
    if "communicator" in msg.lower():
        return True
    return False

def _is_restartable_exc(e: Exception) -> bool:
    # In practice, ML-Agents comm errors can surface like these.
    restartable = (
        _is_timeout_exc(e)
        or _is_unity_env_unloaded_exc(e)
        or isinstance(e, (BrokenPipeError, ConnectionResetError, EOFError))
    )
    return bool(restartable)

def _is_step_after_done_exc(e: Exception) -> bool:
    if UnityGymException is None or not isinstance(e, UnityGymException):
        return False
    return "already returned done = True" in str(e)

