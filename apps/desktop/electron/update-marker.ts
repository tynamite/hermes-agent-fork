/**
 * In-app update mutual-exclusion marker (#50238).
 *
 * The Tauri updater writes HERMES_HOME/.hermes-update-in-progress for the whole
 * duration of an `--update` run (see apps/bootstrap-installer/src-tauri/src/
 * update.rs `UpdateMarkerGuard`). The marker starts with the updater's pid and
 * unix start time. Python may append a third `runtime-restarts` phase line and
 * a fourth process-start identity; legacy readers intentionally ignore extra
 * lines.
 * A short-lived `.hermes-update-in-progress.lock` sidecar serializes stale-
 * marker cleanup with replacement claims across the Python, Rust, and Electron
 * writers. Its owner file also carries a process-start identity, so a
 * suspended live updater remains authoritative without trusting a recycled PID.
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
import { execFileSync } from 'node:child_process'
import path from 'path'

// Legacy markers without a process-start identity use this age ceiling. New
// markers retain a verified live owner past the ceiling while rejecting a
// recycled PID.
export const UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1000
const MARKER_OPERATION_LOCK_STALE_MS = 30 * 1000

/**
 * Return a stable per-process start identity when the host exposes one.
 *
 * Sidecar readers must not expire a suspended live updater. Pairing the PID
 * with this identity preserves that safety while rejecting a recycled PID.
 * The helper intentionally uses only OS facilities available to the desktop
 * process; an unavailable probe is treated conservatively by the caller.
 */
export function getProcessStartIdentity(pid: number): string | null {
  if (!Number.isInteger(pid) || pid <= 0) {
    return null
  }

  if (process.platform === 'linux') {
    try {
      const raw = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')
      const fields = raw.slice(raw.lastIndexOf(')') + 1).trim().split(/\s+/)

      // After the command name, field 22 (starttime) is index 19.
      return fields[19] || null
    } catch {
      return null
    }
  }

  if (process.platform !== 'win32') {
    try {
      const output = execFileSync(
        'ps',
        ['-o', 'lstart=', '-p', String(pid)],
        {
          encoding: 'utf8',
          env: { ...process.env, LC_ALL: 'C' },
          stdio: ['ignore', 'pipe', 'ignore']
        }
      )

      return output.trim() || null
    } catch {
      return null
    }
  }

  try {
    const output = execFileSync(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToFileTimeUtc()`
      ],
      {
        encoding: 'utf8',
        timeout: 2_000,
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'ignore']
      }
    )

    return output.trim() || null
  } catch {
    return null
  }
}

function markerBody(pid: number, startedAt: number, phase = '') {
  const identity = getProcessStartIdentity(pid)

  if (identity) {
    return `${pid}\n${startedAt}\n${phase}\n${identity}\n`
  }

  if (phase) {
    return `${pid}\n${startedAt}\n${phase}\n`
  }

  return `${pid}\n${startedAt}\n`
}

export function markerPath(hermesHome) {
  return path.join(hermesHome, '.hermes-update-in-progress')
}

function markerOperationLockPath(file) {
  return path.join(path.dirname(file), '.hermes-update-in-progress.lock')
}

function isPidZombie(pid) {
  if (process.platform === 'win32') {
    return false
  }

  if (process.platform === 'linux') {
    try {
      const raw = fs.readFileSync(`/proc/${pid}/stat`, 'utf8')
      const fields = raw.slice(raw.lastIndexOf(')') + 1).trim().split(/\s+/)

      return fields[0] === 'Z'
    } catch {
      return false
    }
  }

  try {
    const output = execFileSync(
      'ps',
      ['-o', 'state=', '-p', String(pid)],
      {
        encoding: 'utf8',
        env: { ...process.env, LC_ALL: 'C' },
        stdio: ['ignore', 'pipe', 'ignore']
      }
    )

    return output.trim().startsWith('Z')
  } catch {
    return false
  }
}

function reapStaleMarkerOperationLock(lockDir) {
  const ownerFile = path.join(lockDir, 'owner')
  let ownerPid = null
  let ownerIdentity = null
  let ownerIsMalformed = false

  try {
    const [pidLine, , identityLine] = fs.readFileSync(ownerFile, 'utf8').split('\n')
    const parsedPid = Number.parseInt((pidLine || '').trim(), 10)
    ownerPid = Number.isInteger(parsedPid) && parsedPid > 0 ? parsedPid : null
    ownerIdentity = (identityLine || '').trim() || null
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

  if (ownerPid !== null && isPidAlive(ownerPid)) {
    if (ownerIdentity) {
      const currentIdentity = getProcessStartIdentity(ownerPid)

      // Keep the sidecar when identity inspection is unavailable; opening a
      // concurrent-update window is worse than conservatively waiting.
      if (currentIdentity === null || currentIdentity === ownerIdentity) {
        return false
      }
    } else if (lockAgeMs < MARKER_OPERATION_LOCK_STALE_MS) {
      return false
    }
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
  let ownerIdentity = null

  try {
    const [pidLine, , identityLine] = fs.readFileSync(path.join(lockDir, 'owner'), 'utf8').split('\n')
    const parsedPid = Number.parseInt((pidLine || '').trim(), 10)
    ownerPid = Number.isInteger(parsedPid) && parsedPid > 0 ? parsedPid : null
    ownerIdentity = (identityLine || '').trim() || null
  } catch {
    // The owner file is written immediately after mkdir. Treat that tiny
    // interval as active, but only until the stale-lock ceiling expires.
  }

  const ageMs = Math.max(0, now() - lockStat.mtimeMs)

  if (ownerPid !== null) {
    if (!isPidAlive(ownerPid, kill)) {
      return null
    }

    if (ownerIdentity) {
      const currentIdentity = getProcessStartIdentity(ownerPid)

      if (currentIdentity === null || currentIdentity === ownerIdentity) {
        return { pid: ownerPid, ageMs }
      }

      return null
    }

    return ageMs < MARKER_OPERATION_LOCK_STALE_MS ? { pid: ownerPid, ageMs } : null
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
      const identity = getProcessStartIdentity(process.pid) || ''
      fs.writeFileSync(
        ownerFile,
        `${process.pid}\n${Math.floor(Date.now() / 1000)}\n${identity}\n`,
        {
          encoding: 'utf8',
          flag: 'wx',
          mode: 0o644
        }
      )

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
  // The sidecar is held by the caller, so an exclusive direct create is a
  // no-clobber publication that also works on filesystems without hard-link
  // support (FAT/exFAT and some network mounts).
  fs.writeFileSync(file, body, {
    encoding: 'utf8',
    flag: 'wx',
    mode: 0o644
  })
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

function publishOrReplaceMarkerLocked(file, body, now = Date.now) {
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      publishExclusive(file, body)

      return true
    } catch (err) {
      if (!err || err.code !== 'EEXIST') {
        throw err
      }
    }

    let existingRaw

    try {
      existingRaw = fs.readFileSync(file, 'utf8')
    } catch {
      continue
    }

    const existingLines = String(existingRaw).split('\n')
    const [pidLine, startedLine] = existingLines
    const existingPid = Number.parseInt((pidLine || '').trim(), 10)
    const startedAt = Number.parseInt((startedLine || '').trim(), 10)
    const existingIdentity = (existingLines[3] || '').trim() || null

    // A live claim belongs to another updater (or to a handoff that already
    // published the child pid). Preserve it instead of clobbering it.
    if (markerOwnerIsLive(existingPid, startedAt, existingIdentity, {
      now,
      maxAgeMs: UPDATE_MARKER_MAX_AGE_MS
    })) {
      return false
    }

    // The sidecar is held by the caller, so compare-and-delete the stale
    // contents and retry exclusive publication before releasing the sidecar.
    if (!reclaimStaleMarkerLocked(file, existingRaw)) {
      continue
    }
  }

  // Preserve the historical best-effort contract if a racing writer keeps
  // the path occupied; the caller will leave the existing claim untouched.
  throw Object.assign(new Error('could not publish update marker'), { code: 'EEXIST' })
}

function hasLiveMarkerClaimLocked(file, now) {
  let raw

  try {
    raw = fs.readFileSync(file, 'utf8')
  } catch {
    return false
  }

  const lines = String(raw).split('\n')
  const [pidLine, startedLine] = lines
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const identity = (lines[3] || '').trim() || null

  return markerOwnerIsLive(pid, startedAt, identity, {
    now,
    maxAgeMs: UPDATE_MARKER_MAX_AGE_MS
  })
}

/**
 * Spawn an updater only while holding the shared sidecar, and publish its
 * child PID before releasing that sidecar. A live foreign marker prevents the
 * spawn entirely; if a legacy writer races the claim, the child is terminated
 * before this helper returns.
 */
export function spawnUpdaterWithMarker(hermesHome, spawn, { now = Date.now } = {}) {
  const file = markerPath(hermesHome)
  let child
  let returned = false

  try {
    const result = withMarkerOperationLock(file, () => {
      if (hasLiveMarkerClaimLocked(file, now)) {
        return null
      }

      child = spawn()

      if (!child || !Number.isInteger(child.pid) || child.pid <= 0) {
        return null
      }

      const startedAt = Math.floor(now() / 1000)

      if (!publishOrReplaceMarkerLocked(file, markerBody(child.pid, startedAt), now)) {
        return null
      }

      return child
    })

    if (result.acquired && result.value) {
      returned = true

      return result.value
    }
  } catch {
    // Handoff callers fail closed when the shared claim cannot be acquired or
    // published; an unmarked detached updater must never mutate the checkout.
  } finally {
    if (!returned && child && typeof child.kill === 'function') {
      try {
        child.kill()
      } catch {
        void 0
      }
    }
  }

  return null
}

// True only if a host process with this pid is currently alive. Signal 0 does
// not deliver a signal — it just probes existence/permission. ESRCH => dead;
// EPERM => alive but owned by another user (still "alive" for our purposes).
// Injectable `kill` keeps it unit-testable.
export function isPidAlive(pid, kill: typeof process.kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false
  }

  if (isPidZombie(pid)) {
    return false
  }

  try {
    kill(pid, 0)

    return true
  } catch (err) {
    return Boolean(err && err.code === 'EPERM')
  }
}

function markerOwnerIsLive(
  pid: number,
  startedAt: number,
  identity: string | null,
  { now, maxAgeMs, kill }: {
    now: () => number
    maxAgeMs: number
    kill?: typeof process.kill
  }
) {
  if (!Number.isInteger(pid) || !isPidAlive(pid, kill)) {
    return false
  }

  if (identity) {
    const currentIdentity = getProcessStartIdentity(pid)

    // An unavailable identity probe is not evidence of PID reuse. Retain the
    // claim conservatively rather than opening a concurrent-update window.
    return currentIdentity === null || currentIdentity === identity
  }

  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1000 : Infinity

  return ageMs <= maxAgeMs
}

/**
 * Read + interpret the marker.
 *
 * Returns `{ pid, ageMs }` when an update is GENUINELY still running
 * (parseable pid that is alive, with a matching identity or within the legacy
 * age ceiling), or while the marker
 * operation sidecar is held during publication. Returns `null` for every
 * other "no live update" case — absent, unreadable, malformed, dead pid,
 * recycled identity, or a legacy marker past the ceiling — and, when a stale
 * marker file exists, deletes it so it cannot strand future launches.
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

  const lines = String(raw).split('\n')
  const [pidLine, startedLine] = lines
  const pid = Number.parseInt((pidLine || '').trim(), 10)
  const startedAt = Number.parseInt((startedLine || '').trim(), 10)
  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1000 : Infinity
  const identity = (lines[3] || '').trim() || null

  const alive = markerOwnerIsLive(pid, startedAt, identity, {
    now,
    maxAgeMs,
    kill
  })

  if (!alive) {
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
      // Publish with an exclusive create while the sidecar is held. Readers
      // treat that sidecar as an active operation until the complete body is
      // written, so no hard-link capability is required.
      publishOrReplaceMarkerLocked(file, markerBody(pid, startedAt), now)
    })
  } catch {
    // Best-effort: if we can't write the marker, proceed anyway. The
    // updater will write its own when it reaches run_update.
  }
}
