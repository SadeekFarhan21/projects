import * as vscode from 'vscode'

let bar: vscode.StatusBarItem

export function createStatusBar(context: vscode.ExtensionContext): void {
  bar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10)
  bar.text = '$(clock) WakaFree'
  bar.tooltip = 'Click to open WakaFree menu'
  bar.command = 'wakafree.menu'
  bar.show()
  context.subscriptions.push(bar)
}

export function setStatusBarOk(todayText?: string): void {
  bar.text = todayText ? `$(clock) ${todayText}` : '$(clock) WakaFree'
  bar.tooltip = 'Click to open WakaFree menu'
}

export function setStatusBarError(message: string): void {
  bar.text = '$(warning) WakaFree'
  bar.tooltip = `WakaFree error: ${message} — click for options`
}
