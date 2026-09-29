import ast, sys, re
src = sys.argv[1]
# servo: duty -> pulse width at 50 Hz, constants read from servo_control.py
t = open(f"{src}/hardware/servo_control.py").read()
c = {k: float(re.search(rf"^{k}\s*=\s*([\d.]+)", t, re.M).group(1)) for k in ("MIN_DUTY","MID_DUTY","MAX_DUTY")}
print("servo (constants parsed from hardware/servo_control.py, PWM 50 Hz => period 20 ms)")
for k, d in c.items():
    print(f"  {k}={d}% -> pulse {d/100*20:.2f} ms")
# buzzer: parse fur_elise list
b = open(f"{src}/hardware/buzzer.py").read()
m = re.search(r"fur_elise = (\[.*?\n\])", b, re.S)
mel = ast.literal_eval(re.sub(r"^\s*#.*|\s#\s.*", "", m.group(1), flags=re.M))
notes = [x for x in mel if x[0] != "REST"]
rests = [x for x in mel if x[0] == "REST"]
play = sum(d for _, d in mel)
print("buzzer (parsed from hardware/buzzer.py)")
print(f"  entries={len(mel)} notes={len(notes)} rests={len(rests)}")
print(f"  note+rest durations sum={play:.1f} s; plus 0.1 s default rest after each entry = {play+0.1*len(mel):.1f} s")
