#!/usr/bin/env python3
"""Set each post's front-matter `date:`, `code:` and `tab_title:` from tools/post-dates.json.

A post is dated to when its project started: the first real commit, or, for a
project that lived only locally before it was committed, the creation time of
its earliest file. The manifest records the date and where it came from; this
script writes it into source/_posts and source/_drafts, replacing any existing
`date:` line or adding one after `title:`. Safe to run repeatedly.
"""
import json
import pathlib
import re
import sys

SITE = pathlib.Path(__file__).resolve().parent.parent
DATES = json.loads((SITE / "tools" / "post-dates.json").read_text())


def main() -> int:
    changed, missing = 0, []
    for folder in ("_posts", "_drafts"):
        for post in sorted((SITE / "source" / folder).glob("*.md")):
            entry = DATES.get(post.stem)
            if not entry:
                missing.append(f"{folder}/{post.name}")
                continue
            text = post.read_text()
            m = re.match(r"---\n(.*?)\n---\n", text, re.S)
            if not m:
                missing.append(f"{folder}/{post.name} (no front matter)")
                continue
            new_head = head = m.group(1)
            for key in ("date", "code", "tab_title"):
                if key not in entry:
                    continue
                line = f"{key}: {entry[key]}"
                if re.search(rf"^{key}:.*$", new_head, re.M):
                    new_head = re.sub(rf"^{key}:.*$", line, new_head, count=1, flags=re.M)
                else:
                    new_head = re.sub(r"^(title:.*)$", r"\1\n" + line, new_head, count=1, flags=re.M)
            if new_head != head:
                post.write_text(f"---\n{new_head}\n---\n" + text[m.end():])
                changed += 1
    print(f"dated {changed} posts")
    if missing:
        print("no date in tools/post-dates.json for:", ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
