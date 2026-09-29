import * as vscode from 'vscode'

function cfg() {
  return vscode.workspace.getConfiguration('wakafree')
}

export function getApiUrl(): string {
  return cfg().get<string>('apiUrl', 'http://localhost:3000/api/v1')
}

export function getApiKey(): string {
  return cfg().get<string>('apiKey', '')
}

export function isEnabled(): boolean {
  return cfg().get<boolean>('enabled', true)
}
