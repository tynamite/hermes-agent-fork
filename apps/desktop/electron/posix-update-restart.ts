export interface PosixBackendRestartLatch {
  quiesce<T>(action: () => Promise<T>): Promise<T>
  suppressRestart(): void
  shouldRestart(): boolean
}

/**
 * Arm backend recovery only after every managed child is confirmed stopped.
 * A rejected quiescence leaves the latch disarmed because the old child may
 * still be authoritative; spawning a replacement would create two backends.
 */
export function createPosixBackendRestartLatch(): PosixBackendRestartLatch {
  let restart = false

  return {
    async quiesce<T>(action: () => Promise<T>): Promise<T> {
      const result = await action()

      restart = true

      return result
    },
    suppressRestart(): void {
      restart = false
    },
    shouldRestart(): boolean {
      return restart
    }
  }
}
