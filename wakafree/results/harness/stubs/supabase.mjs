export const calls = []
export const supabase = { from: (t) => ({ upsert: async (rows, o) => { calls.push({ t, n: rows.length }); return { error: null } } }) }
