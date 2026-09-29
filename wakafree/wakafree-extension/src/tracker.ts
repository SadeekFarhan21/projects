import * as vscode from 'vscode'
import * as os from 'os'
import * as path from 'path'
import { execSync } from 'child_process'
import { HeartbeatPayload, sendHeartbeats } from './sender'
import { isEnabled } from './config'
import { setStatusBarOk, setStatusBarError } from './statusbar'

const DEBOUNCE_MS = 2 * 60 * 1000  // only re-send for the same file after 2 min
const FLUSH_MS = 30 * 1000          // batch flush every 30 seconds

const lastSentAt = new Map<string, number>()
const queue: HeartbeatPayload[] = []
let flushTimer: ReturnType<typeof setInterval> | null = null

const LANGUAGE_MAP: Record<string, string> = {
  typescript: 'TypeScript',
  typescriptreact: 'TypeScript JSX',
  javascript: 'JavaScript',
  javascriptreact: 'JavaScript JSX',
  python: 'Python',
  rust: 'Rust',
  go: 'Go',
  java: 'Java',
  c: 'C',
  cpp: 'C++',
  csharp: 'C#',
  ruby: 'Ruby',
  php: 'PHP',
  swift: 'Swift',
  kotlin: 'Kotlin',
  html: 'HTML',
  css: 'CSS',
  scss: 'SCSS',
  less: 'Less',
  json: 'JSON',
  yaml: 'YAML',
  toml: 'TOML',
  markdown: 'Markdown',
  shellscript: 'Shell Script',
  sql: 'SQL',
  dockerfile: 'Dockerfile',
  xml: 'XML',
  vue: 'Vue.js',
  svelte: 'Svelte',
  prisma: 'Prisma',
}

function toLanguageName(id: string): string {
  return LANGUAGE_MAP[id] ?? id
}

function getProject(document: vscode.TextDocument): string | undefined {
  return vscode.workspace.getWorkspaceFolder(document.uri)?.name
}

function getBranch(filePath: string): string | undefined {
  try {
    const branch = execSync('git branch --show-current', {
      cwd: path.dirname(filePath),
      timeout: 1000,
      stdio: ['ignore', 'pipe', 'ignore'],
    })
      .toString()
      .trim()
    return branch || undefined
  } catch {
    return undefined
  }
}

function needsHeartbeat(filePath: string, isWrite: boolean): boolean {
  if (isWrite) return true
  const last = lastSentAt.get(filePath) ?? 0
  return Date.now() - last >= DEBOUNCE_MS
}

export function recordHeartbeat(
  document: vscode.TextDocument,
  isWrite: boolean,
  position?: vscode.Position
): void {
  if (!isEnabled()) return
  if (document.uri.scheme !== 'file') return

  const filePath = document.uri.fsPath
  if (!needsHeartbeat(filePath, isWrite)) return

  lastSentAt.set(filePath, Date.now())

  queue.push({
    entity: filePath,
    type: 'file',
    time: Date.now() / 1000,
    project: getProject(document),
    language: toLanguageName(document.languageId),
    branch: getBranch(filePath),
    is_write: isWrite,
    editor: 'vscode',
    operating_system: `${os.type()} ${os.release()}`,
    machine: os.hostname(),
    lines: document.lineCount,
    lineno: position ? position.line + 1 : undefined,
    cursorpos: position ? position.character + 1 : undefined,
  })
}

async function flush(): Promise<void> {
  if (queue.length === 0) return
  const batch = queue.splice(0, queue.length)
  try {
    await sendHeartbeats(batch)
    setStatusBarOk()
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    setStatusBarError(msg)
    console.error('[WakaFree]', msg)
    // put unsent heartbeats back so they're retried next flush
    queue.unshift(...batch)
  }
}

export function startFlushTimer(): void {
  flushTimer = setInterval(() => {
    flush().catch(console.error)
  }, FLUSH_MS)
}

export function stopFlushTimer(): void {
  if (flushTimer) {
    clearInterval(flushTimer)
    flushTimer = null
  }
  flush().catch(console.error)
}
