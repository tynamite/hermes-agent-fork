import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createPosixBackendRestartLatch } from './posix-update-restart'

test('failed quiescence does not arm backend restart', async () => {
  const latch = createPosixBackendRestartLatch()

  await assert.rejects(
    latch.quiesce(() => Promise.reject(new Error('backend still alive'))),
    /backend still alive/
  )

  assert.equal(latch.shouldRestart(), false)
})

test('verified quiescence arms recovery for later update failure', async () => {
  const latch = createPosixBackendRestartLatch()

  await latch.quiesce(() => Promise.resolve())

  assert.equal(latch.shouldRestart(), true)
})

test('successful handoff suppresses local backend recovery', async () => {
  const latch = createPosixBackendRestartLatch()

  await latch.quiesce(() => Promise.resolve())
  latch.suppressRestart()

  assert.equal(latch.shouldRestart(), false)
})
