# PillifyAI (blog write-up folder)

A Make2025 hackathon prototype for medication management: a doctor prescribes in a mobile app, a server turns the prescription into scheduled dose records, and a physical pill dispenser fetches the due dose, releases it and reports back, so the patient's history shows a device-confirmed dose.

Farhan Sadeek's project, built with teammate Jalen Francis. The canonical repo is the private GitHub repo SadeekFarhan21/PillifyAI. No license file exists, so all rights remain with the authors.

## Why there is no code here

The source repo is private and its history contains credentials (server `.env`, database URLs, Firebase config, WiFi and device API keys). Nothing was copied into this folder. This folder holds only this README and `results/`. Any code excerpts in the post are short, written from files that contain no secrets.

## What the project is

Three parts plus hardware scripts (read from the code; nothing was run):

- `pillifyai/`: Expo ~52 / React Native 0.76.7 app with expo-router, Firebase auth and AsyncStorage. Tabs are Home, History, Account, plus Patients and Prescribe, which `app/(tabs)/_layout.tsx` shows only when `user?.role === 'doctor'`. Registration picks patient (default) or doctor. An offline queue (`services/SyncService.ts`) stores operations in AsyncStorage and replays them on reconnect. Its README is unmodified Expo boilerplate. The only test is the Expo template snapshot test for `ThemedText`.
- `pillifyai-server/`: Express, Sequelize and Postgres, with firebase-admin. Routes are `/api/patients`, `/api/medications`, `/api/auth` and `/api/devices`. Models: User, Doctor, Patient, Medication (frequency enum daily, twice_daily, three_times_daily, weekly, as_needed) and MedicationTracking (status pending, taken, missed or skipped, with deviceId and deviceDispensed). Dispenser endpoints are a ping, listing medications, and a POST to mark a dose dispensed, protected by an `X-API-Key` header or, unverified, any Bearer header, with a hard-coded default key fallback. A large set of repair, reset and integrity scripts suggests data-consistency bugs were hit under time pressure.
- `pillifyai-hardware/`: ESP32 Arduino sketch (307 lines). It connects to WiFi, pings the server, polls every 30 s (`CHECK_INTERVAL = 30000`), uses a 10 s HTTP timeout, and blinks a dispensing LED at 500 ms when a dose is due, then auto-dispenses and POSTs a report. LEDs are on pins 2, 4 and 5. Dispensing is simulated: the sketch has no servo code and the hardware README says the servo mechanism is not implemented in this version.
- `hardware/`: Raspberry Pi Python scripts (RPi.GPIO). `servo_control.py` drives a hacked SG90 on GPIO 2 (physical pin 3) at 50 Hz PWM with duty 2.5, 7.5 and 12.5 for counterclockwise, neutral and clockwise, plus speed by duty offset and a gradual move. `buzzer.py` plays a "Fur Elise" melody (per its own code comment) on a piezo buzzer on GPIO 18. The team note in `ideas.txt` says the SG90 was hacked. Reading the speed-by-duty design as a continuous-rotation conversion is an inference, not stated in the repo.

## Honest limits

- No benchmarks or evaluation exist. This is a working-demo hackathon project, not research.
- No evidence in the repo that the ESP32 and the Pi servo were integrated into one device.
- Hackathon outcome and how it was demoed are unknown and are not claimed.
- Not run: the app needs Firebase config, the server needs Postgres and a `.env`, the hardware needs a board.

## Results in this folder

Everything here is derived from reading or parsing the source; there are no runtime results.

- `results/size_counts.txt`: line counts by part, route handler counts, commit count and dates (34 commits, 2025-03-28 to 2025-03-29). Produced with `wc -l`, `grep -c` and `git log` in the clone; the exact command is not committed, so these counts are not regenerable from this folder.
- `results/servo_buzzer_arithmetic.txt`: produced by `results/measure.py <path-to-clone>`. Parses the servo duty constants and the melody list, and converts duty to pulse width at 50 Hz (0.50, 1.50 and 2.50 ms). The melody has 40 entries (35 notes, 5 rests) summing to 9.2 s of nominal duration, or 13.2 s counting the 0.1 s default gap after each entry. The pulse widths are arithmetic on those constants, not oscilloscope measurements. The melody duration is computed from the note list, not timed on hardware.
