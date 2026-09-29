import * as vscode from 'vscode'
import { recordHeartbeat, startFlushTimer, stopFlushTimer } from './tracker'
import { createStatusBar, setStatusBarOk } from './statusbar'
import { fetchTodayStats } from './sender'

async function refreshStatusBar(): Promise<void> {
  try {
    const text = await fetchTodayStats()
    setStatusBarOk(text)
  } catch {
    // silently ignore — status bar keeps last value
  }
}

function openDashboard(): void {
  const apiUrl: string = vscode.workspace
    .getConfiguration('wakafree')
    .get('apiUrl', 'http://localhost:3000/api/v1')
  const dashboardUrl = apiUrl.replace('/api/v1', '/dashboard')
  vscode.env.openExternal(vscode.Uri.parse(dashboardUrl))
}

export function activate(context: vscode.ExtensionContext): void {
  console.log('[WakaFree] Extension activated')

  createStatusBar(context)
  startFlushTimer()

  // Clicking the status bar opens the dashboard (triggers VS Code's native external URL dialog)
  context.subscriptions.push(
    vscode.commands.registerCommand('wakafree.menu', () => {
      openDashboard()
    })
  )

  // Refresh the status bar with today's time every 5 minutes
  const refreshInterval = setInterval(() => {
    refreshStatusBar().catch(console.error)
  }, 5 * 60 * 1000)
  context.subscriptions.push({ dispose: () => clearInterval(refreshInterval) })

  // Fire a heartbeat when the user switches files
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor((editor) => {
      if (editor?.document) {
        recordHeartbeat(editor.document, false, editor.selection.active)
      }
    })
  )

  // Fire on keystroke activity (debounced internally to 2 min per file)
  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (event.contentChanges.length === 0) return
      const editor = vscode.window.activeTextEditor
      recordHeartbeat(event.document, false, editor?.selection.active)
    })
  )

  // Fire on save — always treated as a write event regardless of debounce
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument((document) => {
      const editor = vscode.window.activeTextEditor
      const pos =
        editor?.document === document ? editor.selection.active : undefined
      recordHeartbeat(document, true, pos)
    })
  )

  // Heartbeat for whatever is open when the extension first loads
  const current = vscode.window.activeTextEditor
  if (current) {
    recordHeartbeat(current.document, false, current.selection.active)
  }

  // Initial status bar refresh
  refreshStatusBar().catch(console.error)
}

export function deactivate(): void {
  stopFlushTimer()
}
