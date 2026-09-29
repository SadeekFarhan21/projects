export const workspace = {
  getConfiguration: () => ({ get: (k, d) => (k === 'apiKey' ? 'test-key' : d) }),
  getWorkspaceFolder: () => ({ name: 'demo' }),
}
export const window = { createStatusBarItem: () => ({ show() {} }) }
export const StatusBarAlignment = { Left: 1, Right: 2 }
