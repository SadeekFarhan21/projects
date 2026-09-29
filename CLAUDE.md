# Writing for this blog

This repo holds Farhan's projects and the Hexo blog in `site/` that writes them up. Posts are about
**real work Farhan built** — never invented projects, results, or numbers.

The writing standard below is adapted from Sean Goedecke, "Writing a tech blog people want to read"
(seangoedecke.com, 2025). `site/source/_posts/building-a-gpt-from-scratch.md` is the house example.

## What makes a post worth reading
- **Have a point.** Each post should carry a clear, specific claim — ideally one a reasonable engineer
  might disagree with, or a result that surprised Farhan. "Unit testing is good" is not a post.
  "Two constants decide silently whether the model trains at all" is.
- **Ideas come from the actual work.** Write about what happened while building the thing: the
  decision, the bug, the measurement that changed the plan. No armchair framing.
- **Skip what's been rehashed** unless the project adds something new to it.
- Returning to the same themes across posts (silent failures, measuring before trusting) is fine;
  a new angle on the same idea is still a new post.

## Titles and framing
- **The title says what Farhan built and its most interesting result or idea**, in plain words.
  Never frame his own project around a flaw, gotcha, or shortcut ("…and Its Fake Timestamps").
  Limitations are honest and belong in Problems / What I Would Change, not the headline or intro.
- The post is written by Farhan about his own work: confident, not self-auditing. It explains what
  was built and what was learned; it does not read like a code review of the repo.
- **Don't write about defects in the code itself.** If something in the project is wrong, stubbed,
  hardcoded, faked, left unrun, or insecure in the backend (placeholder data, estimated timestamps,
  a hardcoded key or threshold, an unverified auth check, a script that never ran), leave it out of
  the post entirely — including titles, summaries, Problems and diagrams. Write about what works and
  the real engineering challenges that were solved. Problems means hard problems overcome while
  building, not a list of the repo's shortcuts.

## Structure for skimmers
- **Lead with a summary as the introduction.** The post opens — with no heading — on a short,
  plain-language introduction of the problem with the core takeaway and a
  high-level technical summary: what the project is, what was found or built, and what it means.
  A non-technical reader should get the point from it alone. Example: "An audit of my 2024 Chicago
  Divvy bike-share analysis found that several of my written claims contradict my own charts, the
  row counts disagree, and parts of the data processing were flawed. Here is what happened and how
  to fix it."
- **Bold the key takeaways.** In Problems and Results especially, put each finding's core takeaway
  in bold (a sentence or clause, not whole paragraphs) so the post can be skimmed by reading only
  the bold text. Use it for findings, not decoration — a few per section, not every line.

## Tone and style
- **Clear and casual.** Plain first person, short sentences, the mechanism explained directly.
  Dan Luu and Paul Graham are the reference points.
- **Few caveats.** Say "the loss is a receipt, not a result", not "in my experience it may be the case
  that…". Readers know it's Farhan's experience. One honest statement of a limitation, in the right
  section, beats hedges sprinkled through every paragraph.
- **Be upfront about the scale of the experience.** State concretely what this was: a weekend
  hackathon build, a class project, a 5M-parameter model on a laptop. Caveat-free tone only works if
  the reader is never misled about scope — so give the real size, time, team, and hardware.
- **Don't narrate the process as evidence.** No commit hashes in prose, no "the git history shows",
  "the deck argues", "the README says". State the decision and its reason. Git is where facts are
  checked, not the story.
- **Credit collaborators plainly by name** for their parts ("Alex built the player"). Forked and team
  repos Farhan worked on are his work to write about; no per-commit accounting.

## Don't write like AI
Adapted from Blake Stockton's "Don't Write Like AI" series (AI Writers Room). These are the tells
readers notice; catch them in editing rather than by prompting around them.
- **Colons.** No colons in titles or headings. Don't use colons in consecutive paragraphs. Never put
  a colon mid-paragraph *and* another before the list that follows it; keep at most one. Prefer a
  real sentence or transition over "The fix is simple: X."
- **Negation.** Avoid "it's not X, it's Y" / "not just X, but Y". State the positive claim.
- **Em dashes.** Don't use them; use a comma, parentheses, or two sentences.
- **Red-flag words and phrases.** Not banned outright (judge in context), but each is a flag and
  a few together read as machine-written. Prefer the specific, plain word. (Blake Stockton, "Don't
  Write Like AI" 4 and 5.)
  - Words: unlock, transform, revolutionize, future-proof, game-changer, strategic, shift, modern,
    today's, real ("real value"), supercharge, harness, leverage, optimize, streamline,
    fundamental(ly), unleash, enhanced, unprecedented, seamless, powerful, intuitive, comprehensive,
    tailored, scalable, agile, dynamic, cutting-edge, best-in-class, next-generation, enable,
    crucial, essential, key (as an adjective), robust, elevate, align, proactive, nuanced,
    innovative, intersection, moreover, thrilled, delve, foster, emphasize.
  - Phrases: "in today's fast-paced…", "in a world where…", "now more than ever", "let's dive in",
    "let's break it down", "here's the thing", "here's what you need to know", "the goal?",
    "the result?", "the good news?", "the bottom line", "that's where X comes in", "it's no secret
    that", "let's face it", "not all X are created equal", "stay ahead of the curve", "imagine a
    world where", "[Problem]? Meet [solution].", "X is more than just Y. It's Z.", "Do X, so you
    can Y."
  - Don't swap in strained second-choice words either ("morphed… nimble… evolving landscape");
    write the plain, specific thing.
- **Vague-change intros.** Never open with "In today's…", "As the [field] continues to evolve…",
  "With the rise of…", "In an increasingly…", "As organizations adapt…" (the formula "As [trend]
  continues to [vague verb], [audience] must [generic goal]"). Write the intro last, once the post
  says what it says, and open on something specific from the project: a number, a concrete moment,
  or a sharp observation. (Blake Stockton, "Don't Write Like AI (6 of 101)".)
- Vary sentence structure; if three paragraphs in a row share a shape, rewrite one.
- **Sentence stacking.** A paragraph of short, standalone factual sentences with the same rhythm
  and no bridges reads like a list without bullets, and readers glaze over. Combine sentences that
  belong together ("Improv builds confidence by helping people think on their feet"), add
  transitions ("so", "that's why", "as a result"), mix short and long sentences, and keep a point of
  view. Pick the few ideas that matter and go deeper on them rather than covering everything.
  Read the paragraph aloud; if it sounds like a list, rewrite it. (Blake Stockton, "Don't Write
  Like AI (2 of 101)".)

## Length and process
- Size the post to the project: a weekend build is roughly 1,200–2,000 words of prose; a research or
  measurement project with real results can run to ~3,000. Cut repetition before substance.
- Drafts live in `site/source/_drafts/` and get at least one more pass before publishing. A draft
  whose idea doesn't hold up when written down stays a draft.
- Every number must come from a file in the project or a command actually run. Never invent results.

## Audience and safety
- Posts go out under Farhan's real name to a public site; the repo is public on GitHub.
- Never publish secrets, credentials, personal data, raw restricted datasets (competition, patient,
  licensed data), or details that could reveal confidential employer/interview material.
- Keep the RSS/Atom feed (`/atom.xml`) and analytics working — readers who like one post should be
  able to follow the rest.

## House format (see existing posts)
Front matter (`layout: post`, Title Case title, 2–4 lowercase topic tags — what the project is
about, never how it was made, so no `team-project`, one-sentence description with the
headline number, and a short `tab_title` for the browser tab: the project's name, or 1–3 words;
record it in `site/tools/post-dates.json`), then these `##` sections in order:

1. **The introduction is the summary — no heading.** The post opens with one or two untitled
   paragraphs that introduce the problem and give the core takeaway and a high-level technical
   summary in plain language (see "Structure for skimmers"). Never write a "## Executive Summary"
   or "## Summary" heading. A `*Reading note.*`, if any, follows it.
2. **Why It Matters** — the problem, who it's for, and why it's worth building or knowing (this is
   what "What I Wanted to Build" used to be).
3. **Technical Details** — the ideas and mechanisms needed to follow the build, with the architecture
   diagrams (what "Theory" and "Architecture" used to be; `###` subsections).
4. **Implementation** — how it was built, with short excerpts of the real code (`###` subsections).
5. Then, as the project warrants: **Problems** (numbered `### 1.` — real challenges solved) ·
   **Experiments** · **Results** · **What I Would Change** · **References**.

No Reproducibility section: Farhan doesn't want build/run instructions in posts. Citations inline as
`<sup>[[N]](#ref-N)</sup>`. Nothing collapsible (no `<details>`). Diagrams only where they explain a
mechanism or setup the prose can't.

**Diagrams must be readable on a laptop.** Posts render diagrams at the text column's width (about
705px on a 1280px screen), so a canvas is scaled by 705 / its width. Keep the Excalidraw canvas at
most ~1,000 units wide with body text at least 18px (titles 28px), which renders at about 12-13px.
Wide, side-by-side layouts become unreadable; stack panels vertically instead.
