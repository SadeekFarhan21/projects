# Arithmetic only, on numbers printed in index.qmd prose and on chart labels (weekday counts).
member, casual = 3127293, 2540693
print("prose total", member+casual, "member%", round(100*member/(member+casual),2), "casual%", round(100*casual/(member+casual),2))
wk = dict(Mon=788188,Tue=803077,Wed=879625,Thu=835692,Fri=841688,Sat=925097,Sun=787201)
print("weekday chart sum", sum(wk.values()), "diff vs prose", sum(wk.values())-(member+casual))
print("min day", min(wk,key=wk.get), "max day", max(wk,key=wk.get))
print("casual/member avg length ratio", round(25.2/12.8,2))
