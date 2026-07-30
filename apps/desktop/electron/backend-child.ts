/**
 * backend-child.ts
 *
 * Windows-aware teardown for the desktop's managed backend child process.
 *
 * Node's `child.kill()` only signals the direct child. On Windows a backend
 * that spawned its own grandchildren (a `hermes` REPL, a pty terminal
 * session, the gateway) survives a plain SIGTERM and keeps files (e.g. the
 * venv shim) locked. So on Windows we tree-kill via `forceKillProcessTree`;
 * everywhere else a plain SIGTERM is correct and sufficient (POSIX has no
 * mandatory locks, and the backend is not spawned detached so there's no
 * process-group to negative-pid-kill).
 *
 * Extracted into its own dependency-free module (no electron import) so the
 * SIGTERM-vs-tree-kill branching can be asserted directly with a fake child
 * object and a spy `forceKillProcessTree`, instead of grepping main.ts source
 * text for the function body.
 */

export interface StopBackendChildDeps {
  /** Defaults to the real platform check; injectable for tests. */
  isWindows?: boolean
  /** Windows tree-kill implementation (real: taskkill /T /F via execFileSync). */
  forceKillProcessTree: (pid: number) => void
}

export interface StopBackendChildrenDeps extends StopBackendChildDeps {
  /** Resolve only after the child has exited or the bounded exit policy ran. */
  waitForExit: (child: KillableChild) => Promise<unknown>
}

export interface KillableChild {
  pid?: number | null
  killed?: boolean
  exitCode?: number | null
  signalCode?: string | null
  kill: (signal: string) => void
}

export interface ConfirmBackendExitDeps {
  isProcessAlive: (pid: number) => boolean
  sleep: (delayMs: number) => Promise<void>
}

/**
 * Stop a managed child process, choosing the right strategy for the platform.
 * No-ops silently if `child` is falsy, already killed, or the kill attempt
 * throws (the process may already be gone) -- mirrors the original inline
 * best-effort semantics in main.ts.
 */
export function stopBackendChild(child: KillableChild | null | undefined, deps: StopBackendChildDeps) {
  if (!child || child.killed) {
    return
  }

  const isWindows = deps.isWindows ?? process.platform === 'win32'

  try {
    if (isWindows && Number.isInteger(child.pid)) {
      deps.forceKillProcessTree(child.pid as number)
    } else {
      child.kill('SIGTERM')
    }
  } catch {
    // Already gone.
  }
}

/**
 * Stop every distinct desktop-managed backend and await its bounded exit path.
 *
 * Update callers must close the whole primary + profile-pool set before
 * mutating the shared Python environment. Deduplication matters during
 * connection handoffs, where two registries can briefly reference one child.
 */
export async function stopBackendChildrenAndWait(
  children: Array<KillableChild | null | undefined>,
  deps: StopBackendChildrenDeps
): Promise<number> {
  const managed = [...new Set(children.filter((child): child is KillableChild => Boolean(child)))]

  for (const child of managed) {
    stopBackendChild(child, deps)
  }

  await Promise.all(managed.map(child => deps.waitForExit(child)))

  return managed.length
}

/**
 * Confirm that a force-killed backend is no longer live.
 *
 * A signal being accepted is not an exit boundary. Poll both the child state
 * and the OS PID table for a short bounded grace, returning false when exit
 * cannot be established so callers can fail closed before mutating files.
 */
export async function confirmBackendExit(
  child: KillableChild,
  deps: ConfirmBackendExitDeps,
  { attempts = 20, intervalMs = 50 } = {}
): Promise<boolean> {
  const pid = child.pid

  for (let attempt = 0; attempt <= attempts; attempt += 1) {
    if (child.exitCode != null || child.signalCode != null) {
      return true
    }

    if (Number.isInteger(pid) && !deps.isProcessAlive(pid as number)) {
      return true
    }

    if (attempt < attempts) {
      await deps.sleep(intervalMs)
    }
  }

  return false
}
