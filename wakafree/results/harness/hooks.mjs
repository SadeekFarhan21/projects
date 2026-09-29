import { pathToFileURL } from 'node:url'
import { existsSync } from 'node:fs'
import path from 'node:path'
const P = process.env.WF
export async function resolve(spec, ctx, next) {
  if (spec === 'vscode') return { url: pathToFileURL(path.resolve('stubs/vscode.mjs')).href, shortCircuit: true }
  if (spec === '@/lib/supabase') return { url: pathToFileURL(path.resolve('stubs/supabase.mjs')).href, shortCircuit: true }
  if (spec.startsWith('@/')) spec = path.join(P, 'wakafree-server/src', spec.slice(2)) + '.ts'
  else if (spec.startsWith('.') && !path.extname(spec) && ctx.parentURL) {
    const abs = path.resolve(path.dirname(new URL(ctx.parentURL).pathname), spec)
    if (existsSync(abs + '.ts')) spec = abs + '.ts'
  }
  if (spec.startsWith('/')) spec = pathToFileURL(spec).href
  return next(spec, ctx)
}
