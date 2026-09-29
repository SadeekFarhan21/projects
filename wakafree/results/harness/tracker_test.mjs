let now = 1_700_000_000_000
Date.now = () => now
const sent = []
globalThis.fetch = async (url, opts) => { sent.push({ url, n: JSON.parse(opts.body).length, bytes: opts.body.length }); return { ok: true } }
const t = await import('./ext/tracker.ts')
const doc = (f) => ({ uri: { scheme: 'file', fsPath: f }, languageId: 'typescript', lineCount: 100 })
const sb = await import('./ext/statusbar.ts'); sb.createStatusBar({ subscriptions: [] })
const out = {}
// 1. debounce: 100 edits on one file, 1 per second (100 s span)
let before = 0
for (let i = 0; i < 100; i++) { t.recordHeartbeat(doc('/w/a.ts'), false, { line: 1, character: 1 }); now += 1000 }
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
out.edits_1s_apart_100s_single_file_heartbeats = sent.reduce((a, s) => a + s.n, 0)
// 2. edits over 10 minutes at 1 per second -> heartbeat count
sent.length = 0
now += 3600_000
for (let i = 0; i < 600; i++) { t.recordHeartbeat(doc('/w/b.ts'), false); now += 1000 }
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
out.edits_1s_apart_600s_single_file_heartbeats = sent.reduce((a, s) => a + s.n, 0)
// 3. saves bypass debounce
sent.length = 0
now += 3600_000
for (let i = 0; i < 5; i++) { t.recordHeartbeat(doc('/w/c.ts'), true); now += 1000 }
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
out.five_saves_in_5s_heartbeats = sent.reduce((a, s) => a + s.n, 0)
// 4. alternate between 3 files every 10s for 10 min
sent.length = 0
now += 3600_000
const fs = ['/w/d1.ts', '/w/d2.ts', '/w/d3.ts']
for (let i = 0; i < 60; i++) { t.recordHeartbeat(doc(fs[i % 3]), false); now += 10_000 }
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
out.three_files_alternating_10s_for_600s_heartbeats = sent.reduce((a, s) => a + s.n, 0)
// 5. payload size
sent.length = 0
now += 3600_000
t.recordHeartbeat(doc('/Users/dev/proj/src/app.ts'), true, { line: 10, character: 4 })
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
out.one_heartbeat_batch_body_bytes = sent[0].bytes
console.log(JSON.stringify(out, null, 2))
// 6. failed send is retried
let fail = true, attempts = []
globalThis.fetch = async (u, o) => { attempts.push(JSON.parse(o.body).length); return fail ? { ok: false, status: 500, text: async () => 'x' } : { ok: true } }
now += 3600_000
for (const f of ['/w/e1.ts', '/w/e2.ts']) t.recordHeartbeat(doc(f), false)
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
fail = false
await t.stopFlushTimer(); await new Promise(r => setTimeout(r, 20))
console.log(JSON.stringify({ retry_attempt_batch_sizes: attempts }))
