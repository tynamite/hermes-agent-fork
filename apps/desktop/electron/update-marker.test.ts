/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report a live update only when the
 * updater pid is alive AND the marker is fresh, (b) treat absent/malformed/
 * dead-pid/expired markers as "no live update" so a crashed updater can't
 * strand future launches, and (c) self-heal by deleting a stale marker file.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import os from 'os'
import path from 'path'

import { test } from 'vitest'

import {
  getProcessStartIdentity,
  isPidAlive,
  markerPath,
  readLiveUpdateMarker,
  spawnUpdaterWithMarker,
  UPDATE_MARKER_MAX_AGE_MS,
  writeUpdateMarker
} from './update-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))

  return dir
}

function writeMarker(home, pid, startedAtSec) {
  fs.writeFileSync(markerPath(home), `${pid}\n${startedAtSec}`)
}

const ALIVE: typeof process.kill = () => true // injected kill that "succeeds" => pid alive

const DEAD: typeof process.kill = () => {
  const err = new Error('no such process')

  ;(err as any).code = 'ESRCH'
  throw err
}

test('absent marker => no live update', () => {
  const home = tmpHome('absent')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
})

test('live operation sidecar blocks readers before marker publication', () => {
  const home = tmpHome('operation-sidecar')
  const operationLock = path.join(home, '.hermes-update-in-progress.lock')
  fs.mkdirSync(operationLock)
  fs.writeFileSync(path.join(operationLock, 'owner'), `${process.pid}\n`)

  const res = readLiveUpdateMarker(home, { kill: ALIVE })

  assert.ok(res, 'a live sidecar is an in-flight update even before marker publish')
  assert.equal(res.pid, process.pid)
})

test('old live-pid sidecar is bounded by the stale ceiling', () => {
  const home = tmpHome('old-operation-sidecar')
  const operationLock = path.join(home, '.hermes-update-in-progress.lock')
  fs.mkdirSync(operationLock)
  fs.writeFileSync(path.join(operationLock, 'owner'), `${process.pid}\n`)
  const old = fs.statSync(operationLock).mtimeMs
  const now = old + 30 * 1000 + 1

  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
})

test('old live sidecar with matching process identity stays active', () => {
  const identity = getProcessStartIdentity(process.pid)

  if (!identity) {
    return
  }

  const home = tmpHome('old-live-identity-sidecar')
  const operationLock = path.join(home, '.hermes-update-in-progress.lock')
  fs.mkdirSync(operationLock)
  fs.writeFileSync(path.join(operationLock, 'owner'), `${process.pid}\n0\n${identity}\n`)
  const old = fs.statSync(operationLock).mtimeMs
  const now = old + 60 * 60 * 1000

  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })

  assert.ok(res, 'a verified live owner remains active after the age ceiling')
  assert.equal(res.pid, process.pid)
})

test('recycled live-pid sidecar is not treated as active', () => {
  if (!getProcessStartIdentity(process.pid)) {
    return
  }

  const home = tmpHome('recycled-live-identity-sidecar')
  const operationLock = path.join(home, '.hermes-update-in-progress.lock')
  fs.mkdirSync(operationLock)
  fs.writeFileSync(path.join(operationLock, 'owner'), `${process.pid}\n0\nold-start\n`)

  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
})

test('live pid within age ceiling => live update reported', () => {
  const home = tmpHome('live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5) // 5s old
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'a fresh, alive marker is a live update')
  assert.equal(res.pid, 4242)
  assert.ok(res.ageMs >= 0 && res.ageMs < 10_000)
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('dead pid => no live update and marker is pruned', () => {
  const home = tmpHome('dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(readLiveUpdateMarker(home, { kill: DEAD }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'a dead-pid marker self-heals (deleted)')
})

test('expired marker (past age ceiling) => no live update and pruned', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  // Even though the pid is "alive", the marker is too old to trust.
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'an expired marker self-heals (deleted)')
})

test('malformed marker => no live update and pruned', () => {
  const home = tmpHome('malformed')
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('isPidAlive: own pid is alive, impossible pid is dead', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(-1), false)
  assert.equal(isPidAlive(0), false)
  assert.equal(isPidAlive(NaN), false)
})

test('isPidAlive: EPERM counts as alive (process owned by another user)', () => {
  const eperm = () => {
    const err = new Error('operation not permitted')

    ;(err as any).code = 'EPERM'
    throw err
  }

  assert.equal(isPidAlive(4242, eperm), true)
})

test('writeUpdateMarker writes a marker that readLiveUpdateMarker accepts', () => {
  const home = tmpHome('write')
  const now = 1_000_000_000_000
  writeUpdateMarker(home, 4242, { now: () => now })
  // The marker should be readable and report the same pid.
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'marker written by writeUpdateMarker should be detected as live')
  assert.equal(res.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'marker file should exist after write')
})

test('writeUpdateMarker never overwrites an existing claim', () => {
  const home = tmpHome('write-existing')
  const now = 1_000_000_000_000
  writeMarker(home, process.pid, Math.floor(now / 1000) - 5)

  writeUpdateMarker(home, 2222, { now: () => now })

  assert.equal(
    fs.readFileSync(markerPath(home), 'utf8'),
    `${process.pid}\n${Math.floor(now / 1000) - 5}`
  )
})

test('writeUpdateMarker replaces a stale existing claim before releasing the sidecar', () => {
  const home = tmpHome('write-replace-stale')
  const now = 1_000_000_000_000
  writeMarker(home, 999999, Math.floor(now / 1000) - 5)

  writeUpdateMarker(home, 2222, { now: () => now })

  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), `2222\n${Math.floor(now / 1000)}\n`)
  assert.ok(
    !fs.existsSync(path.join(home, '.hermes-update-in-progress.lock')),
    'the replacement claim is published before releasing the sidecar'
  )
})

test('spawnUpdaterWithMarker refuses a foreign live claim before spawning', () => {
  const home = tmpHome('spawn-foreign-claim')
  const now = 1_000_000_000_000
  writeMarker(home, process.pid, Math.floor(now / 1000) - 5)
  let spawned = false

  const child = spawnUpdaterWithMarker(
    home,
    () => {
      spawned = true

      return { pid: 4242, kill: () => true }
    },
    { now: () => now }
  )

  assert.equal(child, null)
  assert.equal(spawned, false)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), `${process.pid}\n${Math.floor(now / 1000) - 5}`)
})

test('spawnUpdaterWithMarker publishes the child claim before releasing the sidecar', () => {
  const home = tmpHome('spawn-and-claim')
  const now = 1_000_000_000_000

  const child = spawnUpdaterWithMarker(
    home,
    () => ({ pid: 4242, kill: () => true }),
    { now: () => now }
  )

  assert.ok(child)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), `4242\n${Math.floor(now / 1000)}\n`)
  assert.ok(!fs.existsSync(path.join(home, '.hermes-update-in-progress.lock')))
})

test('writeUpdateMarker reclaims a crashed marker-operation lock', () => {
  const home = tmpHome('write-stale-operation-lock')
  const operationLock = path.join(home, '.hermes-update-in-progress.lock')
  fs.mkdirSync(operationLock)
  fs.writeFileSync(path.join(operationLock, 'owner'), '4294967294\n')

  writeUpdateMarker(home, 2222)

  assert.equal(fs.readFileSync(markerPath(home), 'utf8').split('\n')[0], '2222')
  assert.ok(!fs.existsSync(operationLock), 'a dead sidecar owner must not wedge claims')
})

test('writeUpdateMarker is best-effort (no throw on bad path)', () => {
  // A non-existent directory should not throw.
  const badHome = path.join(os.tmpdir(), 'hermes-marker-nonexistent-' + Date.now())
  assert.doesNotThrow(() => writeUpdateMarker(badHome, 4242))
})

test('writeUpdateMarker + dead pid => self-heals on read', () => {
  const home = tmpHome('write-dead')
  writeUpdateMarker(home, 999999, { now: () => Date.now() })
  // PID 999999 is almost certainly not alive.
  const res = readLiveUpdateMarker(home, { kill: DEAD })
  assert.equal(res, null, 'a dead-pid marker from writeUpdateMarker self-heals')
  assert.ok(!fs.existsSync(markerPath(home)), 'marker file is pruned')
})
