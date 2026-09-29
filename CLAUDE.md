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
headline number), then these `##` sections in order:

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
