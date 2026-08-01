/**
 * In-app update mutual-exclusion marker (#50238).
 *
 * The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
 * duration of an `--update` run (see apps/bootstrap-installer/src-tauri/src/
 * update.rs `UpdateMarkerGuard`). The marker starts with the updater's pid and
 * unix start time. Python may append a third `runtime-restarts` phase line;
 * this reader intentionally ignores additional lines.
 * A short-lived `.hermes-update-in-progress.lock` sidecar serializes stale-
 * marker cleanup with replacement claims across the Python, Rust, and Electron
 * writers.
 *
 * Why: if the user relaunches the desktop mid-update — the window vanished with
 * no progress and looks crashed — a fresh instance must NOT spawn its own local
 * backend. That backend re-locks the venv shim, the updater's straggler cleanup
 * (`force_kill_other_hermes`, taskkill /IM hermes.exe) kills it, the launch
 * fails with the 45s "backend didn't come up" timeout, and the user relaunches
 * into the same trap — an infinite respawn/kill loop. The desktop gates local
 * backend startup on this marker and parks until the update finishes.
 *
 * This module holds the PURE, side-effect-light logic (path, pid liveness,
 * parse + staleness) so it is unit-testable without booting Electron. The
 * polling/boot-progress wrapper lives in main.ts where the boot-progress and
 * log sinks are.
 */

import fs from 'fs'
import path from 'path'

// Even with a live-looking PID, never treat a marker older than this as a live
// update. A full update (git pull + pip + desktop rebuild) is minutes, not tens
// of minutes; past this the marker is almost certainly stale (e.g. the OS
// recycled the pid onto an unrelated process), so the gate self-heals.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000
const MARKER_OPERATION_LOCK_STALE_MS = 30 * 1000

export function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

function markerOperationLockPath(file) {
  return path.join(path.dirname(file), '.hermes-update-in-progress.lock')
}

function reapStaleMarkerOperationLock(lockDir) {
  const ownerFile = path.join(lockDir, 'owner')
  let ownerPid = null
  let ownerIsMalformed = false

  try {
    const [pidLine] = fs.readFileSync(ownerFile, 'utf8').split('\n')
    const parsedPid = Number.parseInt((pidLine || '').trim(), 10)
    ownerPid = Number.isInteger(parsedPid) && parsedPid > 0 ? parsedPid : null
    ownerIsMalformed = ownerPid === null
  } catch {
    ownerIsMalformed = true
  }

  let lockAgeMs

  try {
    lockAgeMs = Math.max(0, Date.now() - fs.statSync(lockDir).mtimeMs)
  } catch {
    return true
  }

  if (ownerIsMalformed) {
    if (lockAgeMs < MARKER_OPERATION_LOCK_STALE_MS) {
      return false
    }
  }

  // Sidecar operations are intentionally short-lived. A live PID after the
  // stale ceiling may have been recycled from a crashed owner, so it must not
  // wedge every future update indefinitely.
  if (
    ownerPid !== null &&
    isPidAlive(ownerPid) &&
    lockAgeMs < MARKER_OPERATION_LOCK_STALE_MS
  ) {
    return false
  }

  try {
    fs.unlinkSync(ownerFile)
  } catch {
    void 0
  }

  try {
    fs.rmdirSync(lockDir)

    return true
  } catch (err) {
    return Boolean(err && err.code === 'ENOENT')
  }
}

function liveMarkerOperationLock(file, { kill, now }) {
  const lockDir = markerOperationLockPath(file)
  let lockStat

  try {
    lockStat = fs.statSync(lockDir)
  } catch {
    return null
  }

  let ownerPid = null

  try {
    const [pidLine] = fs.readFileSync(path.join(lockDir, 'owner'), 'utf8').split('\n')
    const parsedPid = Number.parseInt((pidLine || '').trim(), 10)
    ownerPid = Number.isInteger(parsedPid) && parsedPid > 0 ? parsedPid : null
  } catch {
    // The owner file is written immediately after mkdir. Treat that tiny
    // interval as active, but only until the stale-lock ceiling expires.
  }

  const ageMs = Math.max(0, now() - lockStat.mtimeMs)

  if (ownerPid !== null) {
    return ageMs < MARKER_OPERATION_LOCK_STALE_MS && isPidAlive(ownerPid, kill)
      ? { pid: ownerPid, ageMs }
      : null
  }

  return ageMs < MARKER_OPERATION_LOCK_STALE_MS
    ? { pid: -1, ageMs }
    : null
}

function withMarkerOperationLock(file, operation) {
  const lockDir = markerOperationLockPath(file)
  const ownerFile = path.join(lockDir, 'owner')

  // Marker operations are intentionally short. Avoid blocking Electron's
  // event loop while another process holds the sidecar; the caller can retry
  // or re-read the marker when acquisition loses.
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      fs.mkdirSync(lockDir)
    } catch (err) {
      if (err && err.code === 'EEXIST' && reapStaleMarkerOperationLock(lockDir)) {
        continue
      }

      return { acquired: false, value: undefined }
    }

    try {
      fs.writeFileSync(ownerFile, `${process.pid}\n${Math.floor(Date.now() / 1000)}\n`, {
        encoding: 'utf8',
        flag: 'wx',
        mode: 0o644
      })

      return { acquired: true, value: operation() }
    } finally {
      try {
        fs.unlinkSync(ownerFile)
      } catch {
        void 0
      }

      try {
        fs.rmdirSync(lockDir)
      } catch {
        void 0
      }
    }
  }

  return { acquired: false, value: undefined }
}

function publishExclusive(file, body) {
  const temporary = `${file}.tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`

  try {
    fs.writeFileSync(temporary, body, {
      encoding: 'utf8',
      flag: 'wx',
      mode: 0o644
    })
    fs.linkSync(temporary, file)
  } finally {
    try {
      fs.unlinkSync(temporary)
    } catch {
      void 0
    }
  }
}

function reclaimStaleMarkerLocked(file, expectedRaw) {
  try {
    if (fs.readFileSync(file, 'utf8') !== expectedRaw) {
      return false
    }
  } catch {
    return true
  }

  try {
    fs.unlinkSync(file)

    return true
  } catch {
    return !fs.existsSync(file)
  }
}

function reclaimStaleMarker(file, expectedRaw) {
  try {
    return withMarkerOperationLock(file, () =>
      reclaimStaleMarkerLocked(file, expectedRaw)
    )
  } catch {
    return { acquired: false, value: false }
  }
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
export function isPidAlive(pid, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err) {
    return Boolean(err && err.code === 'EPERM')
  }
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` when an update is GENUINELY still running
 * (parseable pid that is alive, within the age ceiling), or while the marker
 * operation sidecar is held during publication. Returns `null` for every
 * other "no live update" case — absent, unreadable, malformed, dead pid, or
 * past the ceiling — and, when a stale marker file exists, deletes it so it
 * cannot strand future launches.
 *
 * Pure-ish: file I/O against the given path, plus an injectable pid probe and
 * clock for tests.
 */
export function readLiveUpdateMarker(
  hermesHome,
  {
    kill,
    now = Date.now,
    maxAgeMs = UPDATE_MARKER_MAX_AGE_MS,
    _retries = 0
  }: {
    now?: () => number
    maxAgeMs?: number
    kill?: typeof process.kill
    _retries?: number
  } = {}
) {
  const file = markerPath(hermesHome)
  let raw

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    const operation = liveMarkerOperationLock(file, { kill, now })

    if (operation || _retries >= 2) {
      return operation
    }

    // The sidecar may have been released just before the first marker read
    // completed. Re-read once after it appears clear so a newly published live
    // marker cannot be mistaken for absence.
    return readLiveUpdateMarker(hermesHome, {
      kill,
      now,
      maxAgeMs,
      _retries: _retries + 1
    })
  }

  const [pidLine, startedLine] = String(raw).split('\n')
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1000 : Infinity
  const alive = Number.isInteger(pid) && isPidAlive(pid, kill)

  if (!alive || ageMs > maxAgeMs) {
    const expectedRaw = String(raw)
    const reclaimed = reclaimStaleMarker(file, expectedRaw)

    if (!reclaimed.value && _retries < 2) {
      let replacement = expectedRaw

      try {
        replacement = fs.readFileSync(file, 'utf8')
      } catch {
        void 0
      }

      if (replacement !== expectedRaw) {
        return readLiveUpdateMarker(hermesHome, {
          kill,
          now,
          maxAgeMs,
          _retries: _retries + 1
        })
      }
    }

    const operation = liveMarkerOperationLock(file, { kill, now })

    if (operation || _retries >= 2) {
      return operation
    }

    return readLiveUpdateMarker(hermesHome, {
      kill,
      now,
      maxAgeMs,
      _retries: _retries + 1
    })
  }

  return { pid, ageMs }
}

/**
 * Write the update-in-progress marker *from the desktop* before handing off
 * to the detached updater.
 *
 * The Tauri-based hermes-setup.exe takes several seconds to initialise its
 * window and reach the Rust `run_update` entry point where it writes the
 * marker itself. During that gap the desktop's `app.quit()` teardown kills
 * the backend child, the renderer's WebSocket drops, and the renderer
 * immediately calls `ensureBackend()` → `waitForUpdateToFinish()`. Because
 * the updater hasn't written the marker yet, the gate sees no live update
 * and spawns a *new* backend — which re-locks `.pyd` files in the venv.
 * When the updater finally reaches the venv-rebuild stage it finds those
 * files locked and the update bricks.
 *
 * Fix: the desktop writes the marker itself, using the spawned updater's
 * PID, immediately after `spawn()`. The updater's `UpdateMarkerGuard` adopts
 * that pre-claim rather than overwriting it; the marker body is the same
 * format and `readLiveUpdateMarker` only cares that *some* live pid owns it.
 * When the updater finishes it deletes the marker as before.
 * If the updater never starts (spawn failure) the marker still contains a
 * real PID, so `readLiveUpdateMarker` will self-heal once that PID exits.
 */
export function writeUpdateMarker(hermesHome, pid, { now = Date.now } = {}) {
  const file = markerPath(hermesHome)
  const startedAt = Math.floor(now() / 1000)

  try {
    withMarkerOperationLock(file, () => {
      // Fully write a private file, then publish it with a no-clobber hard
      // link. This keeps readers from observing a partial marker and never
      // overwrites a claim that won the sidecar operation lock first.
      publishExclusive(file, `${pid}\n${startedAt}\n`)
    })
  } catch {
    // Best-effort: if we can't write the marker, proceed anyway. The
    // updater will write its own when it reaches run_update.
  }
}
