---
layout: post
title: "A Pill Dispenser Built Around One Dose Record"
tab_title: PillifyAI
code: https://github.com/SadeekFarhan21/projects/tree/main/pillifyai
date: 2025-03-28 22:45:01
tags:
  - hardware
  - iot
  - mobile
description: >-
  A Make2025 hackathon build in about 19.5 hours: a prescribing app, a Node
  server and an ESP32 dispenser that polls every 30 seconds, all sharing one
  dose record.
---

A prescription is a piece of paper and a promise. Nobody can tell whether the pill actually came out of the bottle. At the Make2025 hackathon, Jalen Francis and I built PillifyAI in about 19.5 hours to close that loop. A doctor prescribes in a mobile app. A server turns the prescription into scheduled dose records. A dispenser asks the server what's due, releases the dose, and reports back, so the patient's history can tell a dose the device confirmed from one the patient only claimed.

What made it work is one decision. **A dose is a single row, and every part of the system reads and writes that row.** The app, the server and the dispenser never talk to each other directly, and the row keeps "the patient says they took it" separate from "the device says it released it". Under the hood there's an Expo and React Native app, a Node, Express and Postgres server, an ESP32 that polls the server every 30 seconds, and Raspberry Pi scripts that drive a hacked SG90 servo and a piezo buzzer.

## Why It Matters

I wanted the prescription to become data, the schedule to become rows, and the pill to come out of a box that tells the server it did.

For one hackathon, the scope came down to three things:

- A mobile app where a doctor can add patients and prescribe, and a patient can see today's medications and their history.
- A server that owns the schedule and the truth about each dose.
- A dispenser that asks the server what's due and answers with what it did.

This was a hackathon build, not a clinical product. The point was to get the loop working across a phone, a server and a microcontroller in one overnight session, from 22:45 on 28 March to 18:14 on 29 March 2025. The app came to about 5,954 lines of TypeScript and JavaScript, the server about 3,364 lines of JavaScript, the ESP32 sketch 307 lines, and the Pi scripts 383 lines of Python.

## Technical Details

### One Record as the Shared Truth

The whole design comes down to one idea, and it lives in the dose row. The doctor's prescription creates that row as pending, the dispenser's report marks it dispensed, and the patient's history reads it. Since the state lives only in that row, the three clients never need to coordinate, and each one can be built and tested against the server alone. With two people and one night, that mattered.

The `MedicationTracking` model has exactly the fields that idea needs. Condensed from the source, with layout and some `allowNull` options trimmed:

```js
scheduledFor: { type: DataTypes.DATE, allowNull: false },
takenAt:      { type: DataTypes.DATE, allowNull: true },
status: {
  type: DataTypes.ENUM('pending', 'taken', 'missed', 'skipped'),
  defaultValue: 'pending',
},
deviceId:        { type: DataTypes.STRING, allowNull: true },
requestedByApp:  { type: DataTypes.BOOLEAN, defaultValue: false },
deviceDispensed: { type: DataTypes.BOOLEAN, defaultValue: false },
```

The split between `status` and `deviceDispensed` is the design choice I care about most. A patient can mark a dose taken in the app, and the dispenser can separately report that it released one. Because they're separate fields, the history can tell a claim apart from a confirmation.

### Polling Instead of Pushing

A microcontroller on home WiFi is hard to reach from outside, so the dispenser asks the server instead of waiting to be told. Polling on an interval is the simplest design that works under that constraint, and it fits the request-response REST style the server already speaks<sup>[[1]](#ref-1)</sup>. The cost is latency, and the interval sets the limit. With a 30 second poll, a due dose waits at most 30 seconds before the dispenser notices. A pill scheduled for a given minute can wait that long, and the behavior is easy to reason about.

### Hobby Servos and Pulse Width

A standard hobby servo expects a pulse every 20 ms (50 Hz), and the width of that pulse, usually 1 to 2 ms, sets the shaft angle<sup>[[2]](#ref-2)</sup>. A servo modified to rotate continuously reads the same pulse as a speed and direction around a neutral width instead. We hacked an SG90, and our code drives it by speed, so the further the duty is from neutral, the faster it turns.

At 50 Hz the period is 20 ms, so you get the pulse width by multiplying the duty cycle by the period: 2.5 percent is 0.50 ms, 7.5 percent is 1.50 ms and 12.5 percent is 2.50 ms.

### Auth Lives in Firebase, Roles Live in Postgres

Firebase handles credentials. The server keeps its own user, doctor and patient tables. It depends on firebase-admin, whose usual job is verifying Firebase ID tokens<sup>[[3]](#ref-3)</sup>, and it mirrors users into Postgres, where the doctor and patient roles are stored and joined to prescriptions. That's why the auth routes are mostly about keeping users in sync rather than logging them in.

### Architecture

The system is three clients around one server and one database.

<figure class="excal excal-interactive" data-diagram="pillifyai-architecture"><svg class="excal-svg" role="img" aria-label="PillifyAI architecture: a Firebase auth box (email login, doctor or patient role) points to both the Expo and React Native mobile app, whose tabs are Home, History and Account plus doctor-only Patients and Prescribe and which keeps an offlineOperationsQueue in AsyncStorage, and to the Node Express server; the app and the server talk over REST (POST /prescribe, PUT /status/:id); the server exposes /api/auth, /api/patients, /api/medications and /api/devices and holds a Postgres database with the User, Doctor, Patient, Medication and MedicationTracking models; an ESP32 sketch that polls every 30 s, shows its state with LEDs on pins 2, 4 and 5, and blinks, dispenses and reports when a dose is due talks to the server both ways through GET esp32/medications and POST esp32/dispense/:id with an X-API-Key header; a separate box holds the Raspberry Pi scripts (hacked SG90 servo on GPIO 2 at 50 Hz, piezo buzzer on GPIO 18)." version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 1655" width="980" height="1655"><!-- svg-source:excalidraw --><metadata></metadata><defs></defs><rect x="0" y="0" width="980" height="1655" fill="#ffffff"></rect><g transform="translate(10 10) rotate(0 311.64000000000004 17.5)"><text x="0" y="25.060546875" font-family="Quicksand" font-size="28px" fill="#0b4f7c" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="700">PillifyAI: Three Parts Around One Database</text></g><g transform="translate(10 60) rotate(0 319.59000000000003 22.5)"><text x="0" y="16.1103515625" font-family="Quicksand" font-size="18px" fill="#64748b" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">A doctor prescribes in the app, the server turns it into dose rows,</text><text x="0" y="38.6103515625" font-family="Quicksand" font-size="18px" fill="#64748b" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">and a dispenser polls for the due dose and reports back.</text></g><a href="#auth-lives-in-firebase-roles-live-in-postgres" class="excal-hot"><title>Auth Lives in Firebase, Roles Live in Postgres</title><g stroke-linecap="round" transform="translate(10 160) rotate(0 350 45)"><path d="M22.5 0 C217.93 0, 413.37 0, 677.5 0 C692.5 0, 700 7.5, 700 22.5 C700 38.81, 700 55.11, 700 67.5 C700 82.5, 692.5 90, 677.5 90 C533.85 90, 390.21 90, 22.5 90 C7.5 90, 0 82.5, 0 67.5 C0 52.19, 0 36.88, 0 22.5 C0 7.5, 7.5 0, 22.5 0" stroke="none" stroke-width="0" fill="#ebe3f6"></path><path d="M22.5 0 C153.81 0, 285.12 0, 677.5 0 M22.5 0 C257.87 0, 493.25 0, 677.5 0 M677.5 0 C692.5 0, 700 7.5, 700 22.5 M677.5 0 C692.5 0, 700 7.5, 700 22.5 M700 22.5 C700 33.2, 700 43.9, 700 67.5 M700 22.5 C700 35.99, 700 49.48, 700 67.5 M700 67.5 C700 82.5, 692.5 90, 677.5 90 M700 67.5 C700 82.5, 692.5 90, 677.5 90 M677.5 90 C516.63 90, 355.75 90, 22.5 90 M677.5 90 C506.46 90, 335.41 90, 22.5 90 M22.5 90 C7.5 90, 0 82.5, 0 67.5 M22.5 90 C7.5 90, 0 82.5, 0 67.5 M0 67.5 C0 53.33, 0 39.15, 0 22.5 M0 67.5 C0 57.43, 0 47.36, 0 22.5 M0 22.5 C0 7.5, 7.5 0, 22.5 0 M0 22.5 C0 7.5, 7.5 0, 22.5 0" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g></a><g transform="translate(34 182) rotate(0 166.95000000000002 22.5)"><text x="0" y="16.1103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Firebase auth</text><text x="0" y="38.6103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">email login, doctor or patient role</text></g><a href="#roles-in-the-app" class="excal-hot"><title>Roles in the App</title><g stroke-linecap="round" transform="translate(10 330) rotate(0 350 175)"><path d="M32 0 C272.6 0, 513.21 0, 668 0 C689.33 0, 700 10.67, 700 32 C700 107.44, 700 182.89, 700 318 C700 339.33, 689.33 350, 668 350 C438.41 350, 208.81 350, 32 350 C10.67 350, 0 339.33, 0 318 C0 228.42, 0 138.84, 0 32 C0 10.67, 10.67 0, 32 0" stroke="none" stroke-width="0" fill="#fbe3d5"></path><path d="M32 0 C159.51 0, 287.02 0, 668 0 M32 0 C189.3 0, 346.6 0, 668 0 M668 0 C689.33 0, 700 10.67, 700 32 M668 0 C689.33 0, 700 10.67, 700 32 M700 32 C700 120.02, 700 208.03, 700 318 M700 32 C700 105.76, 700 179.52, 700 318 M700 318 C700 339.33, 689.33 350, 668 350 M700 318 C700 339.33, 689.33 350, 668 350 M668 350 C471.92 350, 275.84 350, 32 350 M668 350 C461.67 350, 255.34 350, 32 350 M32 350 C10.67 350, 0 339.33, 0 318 M32 350 C10.67 350, 0 339.33, 0 318 M0 318 C0 243.01, 0 168.01, 0 32 M0 318 C0 223.79, 0 129.57, 0 32 M0 32 C0 10.67, 10.67 0, 32 0 M0 32 C0 10.67, 10.67 0, 32 0" stroke="#c0572a" stroke-width="2" fill="none"></path></g></a><g transform="translate(34 348) rotate(0 180.73000000000002 13.75)"><text x="0" y="19.6904296875" font-family="Quicksand" font-size="22px" fill="#c0572a" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Mobile app (Expo, React Native)</text></g><g transform="translate(34 390) rotate(0 181.26000000000002 67.5)"><text x="0" y="16.1103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Tabs: Home, History, Account</text><text x="0" y="38.6103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Doctor only: Patients, Prescribe</text><text x="0" y="61.1103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Screens: login, register, patient/[id]</text><text x="0" y="83.6103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500"></text><text x="0" y="106.1103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">services/MedicationService.ts</text><text x="0" y="128.6103515625" font-family="Quicksand" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">services/SyncService.ts</text></g><a href="#an-offline-queue" class="excal-hot"><title>An Offline Queue</title><g stroke-linecap="round" transform="translate(34 548) rotate(0 150 53)"><path d="M26.5 0 C100.16 0, 173.82 0, 273.5 0 C291.17 0, 300 8.83, 300 26.5 C300 37.94, 300 49.39, 300 79.5 C300 97.17, 291.17 106, 273.5 106 C181.24 106, 88.98 106, 26.5 106 C8.83 106, 0 97.17, 0 79.5 C0 59.75, 0 40, 0 26.5 C0 8.83, 8.83 0, 26.5 0" stroke="none" stroke-width="0" fill="#1e293b"></path><path d="M26.5 0 C76.03 0, 125.56 0, 273.5 0 M26.5 0 C95.48 0, 164.47 0, 273.5 0 M273.5 0 C291.17 0, 300 8.83, 300 26.5 M273.5 0 C291.17 0, 300 8.83, 300 26.5 M300 26.5 C300 37.78, 300 49.05, 300 79.5 M300 26.5 C300 47.44, 300 68.37, 300 79.5 M300 79.5 C300 97.17, 291.17 106, 273.5 106 M300 79.5 C300 97.17, 291.17 106, 273.5 106 M273.5 106 C223.52 106, 173.55 106, 26.5 106 M273.5 106 C219.32 106, 165.15 106, 26.5 106 M26.5 106 C8.83 106, 0 97.17, 0 79.5 M26.5 106 C8.83 106, 0 97.17, 0 79.5 M0 79.5 C0 59.2, 0 38.89, 0 26.5 M0 79.5 C0 58.94, 0 38.38, 0 26.5 M0 26.5 C0 8.83, 8.83 0, 26.5 0 M0 26.5 C0 8.83, 8.83 0, 26.5 0" stroke="#1e293b" stroke-width="2" fill="none"></path></g></a><g transform="translate(52 565) rotate(0 118.80000000000001 33.75)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#22c55e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">offlineOperationsQueue</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#22c55e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">in AsyncStorage,</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#22c55e" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">replayed on reconnect</text></g><a href="#the-prescribe-handler" class="excal-hot"><title>The Prescribe Handler</title><g stroke-linecap="round" transform="translate(10 810) rotate(0 350 245)"><path d="M32 0 C272.51 0, 513.01 0, 668 0 C689.33 0, 700 10.67, 700 32 C700 167.19, 700 302.38, 700 458 C700 479.33, 689.33 490, 668 490 C467.52 490, 267.05 490, 32 490 C10.67 490, 0 479.33, 0 458 C0 310.76, 0 163.53, 0 32 C0 10.67, 10.67 0, 32 0" stroke="none" stroke-width="0" fill="#e3eef8"></path><path d="M32 0 C159.54 0, 287.08 0, 668 0 M32 0 C265.58 0, 499.16 0, 668 0 M668 0 C689.33 0, 700 10.67, 700 32 M668 0 C689.33 0, 700 10.67, 700 32 M700 32 C700 152.46, 700 272.92, 700 458 M700 32 C700 182.43, 700 332.86, 700 458 M700 458 C700 479.33, 689.33 490, 668 490 M700 458 C700 479.33, 689.33 490, 668 490 M668 490 C499.44 490, 330.89 490, 32 490 M668 490 C488.25 490, 308.51 490, 32 490 M32 490 C10.67 490, 0 479.33, 0 458 M32 490 C10.67 490, 0 479.33, 0 458 M0 458 C0 317.29, 0 176.58, 0 32 M0 458 C0 332.93, 0 207.87, 0 32 M0 32 C0 10.67, 10.67 0, 32 0 M0 32 C0 10.67, 10.67 0, 32 0" stroke="#0b4f7c" stroke-width="2" fill="none"></path></g></a><g transform="translate(34 828) rotate(0 122.43 13.75)"><text x="0" y="19.6904296875" font-family="Quicksand" font-size="22px" fill="#0b4f7c" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Node server (Express)</text></g><g transform="translate(34 870) rotate(0 194.4 67.5)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">/api/auth        sync Firebase users</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">/api/patients</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">/api/medications  today, history,</text><text x="0" y="84.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">                  status, prescribe</text><text x="0" y="107.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">/api/devices      ping, medications,</text><text x="0" y="129.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">                  esp32/dispense/:id</text></g><a href="#one-record-as-the-shared-truth" class="excal-hot"><title>One Record as the Shared Truth</title><g stroke-linecap="round" transform="translate(34 1040) rotate(0 326 118)"><path d="M32 0 C207.26 0, 382.52 0, 620 0 C641.33 0, 652 10.67, 652 32 C652 78.35, 652 124.7, 652 204 C652 225.33, 641.33 236, 620 236 C427.28 236, 234.57 236, 32 236 C10.67 236, 0 225.33, 0 204 C0 168.73, 0 133.46, 0 32 C0 10.67, 10.67 0, 32 0" stroke="none" stroke-width="0" fill="#cfe3f3"></path><path d="M32 0 C149.93 0, 267.86 0, 620 0 M32 0 C266.75 0, 501.49 0, 620 0 M620 0 C641.33 0, 652 10.67, 652 32 M620 0 C641.33 0, 652 10.67, 652 32 M652 32 C652 98.7, 652 165.4, 652 204 M652 32 C652 81.92, 652 131.85, 652 204 M652 204 C652 225.33, 641.33 236, 620 236 M652 204 C652 225.33, 641.33 236, 620 236 M620 236 C408.87 236, 197.75 236, 32 236 M620 236 C398.01 236, 176.01 236, 32 236 M32 236 C10.67 236, 0 225.33, 0 204 M32 236 C10.67 236, 0 225.33, 0 204 M0 204 C0 160.8, 0 117.6, 0 32 M0 204 C0 143.44, 0 82.88, 0 32 M0 32 C0 10.67, 10.67 0, 32 0 M0 32 C0 10.67, 10.67 0, 32 0" stroke="#0b4f7c" stroke-width="2" fill="none"></path></g></a><g transform="translate(54 1056) rotate(0 157.41 13.75)"><text x="0" y="19.6904296875" font-family="Quicksand" font-size="22px" fill="#0b4f7c" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Postgres (Sequelize models)</text></g><g transform="translate(54 1098) rotate(0 167.4 78.75)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">User, Doctor, Patient</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">Medication</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">  name, dosage, frequency, time</text><text x="0" y="84.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">MedicationTracking</text><text x="0" y="107.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">  scheduledFor, takenAt,</text><text x="0" y="129.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">  status, deviceId,</text><text x="0" y="152.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">  deviceDispensed</text></g><a href="#the-esp32-loop" class="excal-hot"><title>The ESP32 Loop</title><g stroke-linecap="round" transform="translate(10 1430) rotate(0 230 107.5)"><path d="M32 0 C118.32 0, 204.63 0, 428 0 C449.33 0, 460 10.67, 460 32 C460 65.46, 460 98.93, 460 183 C460 204.33, 449.33 215, 428 215 C293.25 215, 158.5 215, 32 215 C10.67 215, 0 204.33, 0 183 C0 143.06, 0 103.12, 0 32 C0 10.67, 10.67 0, 32 0" stroke="none" stroke-width="0" fill="#dcefe3"></path><path d="M32 0 C111.43 0, 190.87 0, 428 0 M32 0 C123.55 0, 215.1 0, 428 0 M428 0 C449.33 0, 460 10.67, 460 32 M428 0 C449.33 0, 460 10.67, 460 32 M460 32 C460 76.21, 460 120.43, 460 183 M460 32 C460 66.34, 460 100.67, 460 183 M460 183 C460 204.33, 449.33 215, 428 215 M460 183 C460 204.33, 449.33 215, 428 215 M428 215 C327.77 215, 227.55 215, 32 215 M428 215 C320.1 215, 212.21 215, 32 215 M32 215 C10.67 215, 0 204.33, 0 183 M32 215 C10.67 215, 0 204.33, 0 183 M0 183 C0 126.82, 0 70.65, 0 32 M0 183 C0 151.2, 0 119.4, 0 32 M0 32 C0 10.67, 10.67 0, 32 0 M0 32 C0 10.67, 10.67 0, 32 0" stroke="#2f7a54" stroke-width="2" fill="none"></path></g></a><g transform="translate(34 1448) rotate(0 128.26000000000002 13.75)"><text x="0" y="19.6904296875" font-family="Quicksand" font-size="22px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">ESP32 sketch (Arduino)</text></g><g transform="translate(34 1488) rotate(0 172.79999999999998 56.25)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">WiFi, ping, then poll every 30 s</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">LEDs: pin 2 connected,</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">  pin 4 dispensing, pin 5 error</text><text x="0" y="84.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">Due dose: blink, dispense,</text><text x="0" y="107.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">report to the server</text></g><a href="#the-servo-class-on-the-pi" class="excal-hot"><title>The Servo Class on the Pi</title><g stroke-linecap="round" transform="translate(510 1430) rotate(0 230 107.5)"><path d="M32 0 C165.8 0, 299.6 0, 428 0 C449.33 0, 460 10.67, 460 32 C460 88.44, 460 144.87, 460 183 C460 204.33, 449.33 215, 428 215 C288.29 215, 148.57 215, 32 215 C10.67 215, 0 204.33, 0 183 C0 134.08, 0 85.17, 0 32 C0 10.67, 10.67 0, 32 0" stroke="none" stroke-width="0" fill="#dcefe3"></path><path d="M32 0 C111.44 0, 190.89 0, 428 0 M32 0 C136.21 0, 240.42 0, 428 0 M428 0 C449.33 0, 460 10.67, 460 32 M428 0 C449.33 0, 460 10.67, 460 32 M460 32 C460 92.07, 460 152.14, 460 183 M460 32 C460 87.04, 460 142.08, 460 183 M460 183 C460 204.33, 449.33 215, 428 215 M460 183 C460 204.33, 449.33 215, 428 215 M428 215 C290.54 215, 153.07 215, 32 215 M428 215 C282.52 215, 137.03 215, 32 215 M32 215 C10.67 215, 0 204.33, 0 183 M32 215 C10.67 215, 0 204.33, 0 183 M0 183 C0 138.78, 0 94.55, 0 32 M0 183 C0 142.37, 0 101.74, 0 32 M0 32 C0 10.67, 10.67 0, 32 0 M0 32 C0 10.67, 10.67 0, 32 0" stroke="#2f7a54" stroke-width="2" fill="none"></path></g></a><g transform="translate(534 1448) rotate(0 169.07000000000005 13.75)"><text x="0" y="19.6904296875" font-family="Quicksand" font-size="22px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">Raspberry Pi scripts (Python)</text></g><g transform="translate(534 1488) rotate(0 172.79999999999995 33.75)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">Hacked SG90 servo, GPIO 2,</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">50 Hz PWM, duty 2.5 / 7.5 / 12.5</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#374151" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">Piezo buzzer, GPIO 18</text></g><g stroke-linecap="round"><g transform="translate(185 250) rotate(0 0 40)"><path d="M0 0 C0 16.05, 0 32.1, 0 80 M0 0 C0 23.61, 0 47.22, 0 80" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g><g transform="translate(185 250) rotate(0 0 40)"><path d="M-8.55 56.51 C-6.83 61.22, -5.12 65.93, 0 80 M-8.55 56.51 C-6.03 63.44, -3.5 70.37, 0 80" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g><g transform="translate(185 250) rotate(0 0 40)"><path d="M8.55 56.51 C6.83 61.22, 5.12 65.93, 0 80 M8.55 56.51 C6.03 63.44, 3.5 70.37, 0 80" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g></g><mask></mask><g stroke-linecap="round"><g transform="translate(710 205) rotate(0 60 327)"><path d="M0 0 C24.08 0, 48.16 0, 120 0 M0 0 C28.69 0, 57.38 0, 120 0 M120 0 C120 192.5, 120 385.01, 120 654 M120 0 C120 141.15, 120 282.31, 120 654 M120 654 C93.68 654, 67.35 654, 0 654 M120 654 C80.18 654, 40.36 654, 0 654" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g><g transform="translate(710 205) rotate(0 60 327)"><path d="M23.49 645.45 C18.78 647.17, 14.06 648.88, 0 654 M23.49 645.45 C17.88 647.49, 12.26 649.54, 0 654" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g><g transform="translate(710 205) rotate(0 60 327)"><path d="M23.49 662.55 C18.78 660.83, 14.06 659.12, 0 654 M23.49 662.55 C17.88 660.51, 12.26 658.46, 0 654" stroke="#7d5ba6" stroke-width="2" fill="none"></path></g></g><mask></mask><g transform="translate(850 482.5) rotate(0 28.620000000000005 22.5)"><text x="0" y="16.1103515625" font-family="Quicksand" font-size="18px" fill="#7d5ba6" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">verify</text><text x="0" y="38.6103515625" font-family="Quicksand" font-size="18px" fill="#7d5ba6" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic" font-weight="500">users</text></g><g stroke-linecap="round"><g transform="translate(185 680) rotate(0 0 65)"><path d="M0 0 C0 26.09, 0 52.17, 0 130 M0 0 C0 42.52, 0 85.04, 0 130" stroke="#c0572a" stroke-width="2" fill="none"></path></g><g transform="translate(185 680) rotate(0 0 65)"><path d="M8.55 23.49 C6.83 18.78, 5.12 14.06, 0 0 M8.55 23.49 C5.75 15.81, 2.96 8.12, 0 0" stroke="#c0572a" stroke-width="2" fill="none"></path></g><g transform="translate(185 680) rotate(0 0 65)"><path d="M-8.55 23.49 C-6.83 18.78, -5.12 14.06, 0 0 M-8.55 23.49 C-5.75 15.81, -2.96 8.12, 0 0" stroke="#c0572a" stroke-width="2" fill="none"></path></g><g transform="translate(185 680) rotate(0 0 65)"><path d="M-8.55 106.51 C-6.83 111.22, -5.12 115.94, 0 130 M-8.55 106.51 C-5.75 114.19, -2.96 121.88, 0 130" stroke="#c0572a" stroke-width="2" fill="none"></path></g><g transform="translate(185 680) rotate(0 0 65)"><path d="M8.55 106.51 C6.83 111.22, 5.12 115.94, 0 130 M8.55 106.51 C5.75 114.19, 2.96 121.88, 0 130" stroke="#c0572a" stroke-width="2" fill="none"></path></g></g><mask></mask><g transform="translate(210 711) rotate(0 81 33.75)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#c0572a" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">REST</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#c0572a" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">POST /prescribe</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#c0572a" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">PUT /status/:id</text></g><g stroke-linecap="round"><g transform="translate(125 1430) rotate(0 0 -65)"><path d="M0 0 C0 -26.09, 0 -52.18, 0 -130 M0 0 C0 -27.96, 0 -55.91, 0 -130" stroke="#2f7a54" stroke-width="2" fill="none"></path></g><g transform="translate(125 1430) rotate(0 0 -65)"><path d="M-8.55 -23.49 C-6.83 -18.78, -5.12 -14.06, 0 0 M-8.55 -23.49 C-6.71 -18.44, -4.87 -13.39, 0 0" stroke="#2f7a54" stroke-width="2" fill="none"></path></g><g transform="translate(125 1430) rotate(0 0 -65)"><path d="M8.55 -23.49 C6.83 -18.78, 5.12 -14.06, 0 0 M8.55 -23.49 C6.71 -18.44, 4.87 -13.39, 0 0" stroke="#2f7a54" stroke-width="2" fill="none"></path></g><g transform="translate(125 1430) rotate(0 0 -65)"><path d="M8.55 -106.51 C6.83 -111.22, 5.12 -115.94, 0 -130 M8.55 -106.51 C6.71 -111.56, 4.87 -116.61, 0 -130" stroke="#2f7a54" stroke-width="2" fill="none"></path></g><g transform="translate(125 1430) rotate(0 0 -65)"><path d="M-8.55 -106.51 C-6.83 -111.22, -5.12 -115.94, 0 -130 M-8.55 -106.51 C-6.71 -111.56, -4.87 -116.61, 0 -130" stroke="#2f7a54" stroke-width="2" fill="none"></path></g></g><mask></mask><g transform="translate(155 1320) rotate(0 64.80000000000001 45)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">GET esp32/</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">medications</text><text x="0" y="62.490234375" font-family="Fragment Mono" font-size="18px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">POST esp32/</text><text x="0" y="84.990234375" font-family="Fragment Mono" font-size="18px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">dispense/:id</text></g><g transform="translate(390 1342.5) rotate(0 48.599999999999994 22.5)"><text x="0" y="17.490234375" font-family="Fragment Mono" font-size="18px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">X-API-Key</text><text x="0" y="39.990234375" font-family="Fragment Mono" font-size="18px" fill="#2f7a54" text-anchor="start" style="white-space: pre;" direction="ltr" dominant-baseline="alphabetic">header</text></g></svg><figcaption class="excal-hint">Hover a box to see where it is explained; click to jump there.</figcaption></figure>

The mobile app and the server talk over REST. The app's service layer covers today's medications, history, updating a taken status, and prescribing. The server mounts four route groups under `/api` (`auth`, `patients`, `medications` and `devices`). The device routes are the dispenser's entire interface: a ping, a listing of patients and medications, and a POST that marks a dose dispensed, sent with an `X-API-Key` header.

There are two pieces of hardware code. The ESP32 sketch handles the network side. The Raspberry Pi scripts handle the mechanical side, driving the servo and the buzzer.

One dose moves through the system like this.

<figure class="excal" data-diagram="pillifyai-dose-loop"><a href="/img/diagrams/pillifyai-dose-loop.webp" class="excal-link" aria-label="Open the diagram full size"><img src="/img/diagrams/pillifyai-dose-loop.webp" alt="One dose across three lanes: the doctor prescribes in the app's Prescribe tab (name, dosage, frequency, time); the server writes a Medication row plus scheduled MedicationTracking rows in the pending state; the ESP32 polls GET esp32/medications every 30 s and finds a due dose; its dispensing LED on pin 4 blinks at 500 ms and the sketch auto-dispenses; the ESP32 POSTs esp32/dispense/:id with deviceId and dispensed: true so the server marks the dose dispensed; and the patient's History screen reads the tracking rows for that patient." width="2400" height="3115" loading="lazy" decoding="async"></a></figure>

## Implementation

### Roles in the App

At registration a user picks a role, and patient is the default. The tab layout shows the Patients and Prescribe tabs only when the signed-in user is a doctor, so a patient sees Home, History and Account and a doctor sees five tabs. There are also login, register and `patient/[id]` screens, routed with expo-router<sup>[[4]](#ref-4)</sup>.

### An Offline Queue

A phone isn't always online, and a doctor adding a patient in a clinic corridor shouldn't lose the entry. The app keeps a list of pending operations in AsyncStorage under `offlineOperationsQueue` and replays them in order once the connection test passes. Condensed from the source, with the try/catch, the connectivity test, logging and the empty-queue return trimmed:

```ts
const queue = await AsyncStorage.getItem('offlineOperationsQueue');
if (!queue) return true;
const operations = JSON.parse(queue);
for (const op of operations) {
  switch (op.operation) {
    case 'addPatient':              await processAddPatient(op.data); break;
    case 'updateMedicationStatus':  await processUpdateMedicationStatus(op.data); break;
    case 'prescribeMedication':     await processPrescribeMedication(op.data); break;
  }
}
await AsyncStorage.setItem('offlineOperationsQueue', JSON.stringify([]));
```

The queue replays in order and is cleared once the loop finishes.

### The Prescribe Handler

Prescribing is the largest handler in the server, about 230 lines. It has to create a `Medication` and generate the scheduled tracking rows from the frequency enum (`daily`, `twice_daily`, `three_times_daily`, `weekly`, `as_needed`). This is the handler that turns a prescription into rows. The other medication routes are short: `GET /today/:userId`, `GET /history/:userId` and `PUT /status/:id`.

### The ESP32 Loop

The ESP32 sketch connects to WiFi with retries, pings the server, then polls every 30 seconds (`CHECK_INTERVAL = 30000`). It parses JSON with ArduinoJson<sup>[[5]](#ref-5)</sup> on the Arduino core for the ESP32<sup>[[6]](#ref-6)</sup>, uses a 10 second HTTP timeout, and shares one request helper between calls. Three LEDs show state, with pin 2 for connected, pin 4 for dispensing and pin 5 for an error. When a dose is due, the dispensing LED blinks and the sketch dispenses on its own.

```cpp
void handleMedicationDue() {
  static unsigned long lastBlinkTime = 0;
  static boolean blinkState = false;

  if (millis() - lastBlinkTime > 500) {
    lastBlinkTime = millis();
    blinkState = !blinkState;
    digitalWrite(LED_DISPENSING, blinkState);
  }

  static int blinkCount = 0;
  if (blinkState && ++blinkCount > 10) {  // After 5 seconds (10 blinks)
    dispenseMedication();
    blinkCount = 0;
  }
}
```

`dispenseMedication` holds for a second, reports the dispense to the server, and clears its state. The sketch also has a `remoteDispense(trackingId)` function for a dispense the server triggers.

### The Servo Class on the Pi

The Raspberry Pi scripts use RPi.GPIO<sup>[[7]](#ref-7)</sup>. A `ServoControl` class starts 50 Hz PWM on GPIO 2 (physical pin 3) at the neutral duty and gives three ways to move, `set_direction`, `set_speed` and `gradual_move`. Condensed from the source, with comments changed or trimmed and the else branch and the rest of `set_speed` cut:

```python
MIN_DUTY = 2.5   # fully counterclockwise
MID_DUTY = 7.5   # neutral
MAX_DUTY = 12.5  # fully clockwise

def set_speed(self, speed_percent):
    if self.current_duty < MID_DUTY:      # counterclockwise
        range_width = MID_DUTY - MIN_DUTY
        duty = MID_DUTY - (range_width * speed_percent / 100)
    elif self.current_duty > MID_DUTY:    # clockwise
        range_width = MAX_DUTY - MID_DUTY
        duty = MID_DUTY + (range_width * speed_percent / 100)
```

Speed is a percentage of the distance from neutral to the end of the range, in whichever direction the servo is already turning. Other scripts include `servo_examples.py`, `slow_counterclockwise.py` and a CLI menu.

### The Buzzer

`buzzer.py` drives a piezo buzzer on GPIO 18 through PWM, starting at 1 kHz. It has a dictionary from note names to frequencies and a `play_note(note, duration)` function, and it plays Für Elise.

## Problems

### 1. Two Systems Holding One Identity

Firebase knows who a user is. Postgres knows their role and their patients. **The two have to agree, and most of the auth code exists to keep them in step.** The auth routes handle registration, syncing a Firebase user into Postgres (`sync-firebase-user`), checking whether a user exists, looking one up by email, updating a role, and making sure a doctor record exists (`ensure-doctor-record`). **Splitting it this way let Firebase own credentials while every prescription could still join against a real doctor and patient row.**

### 2. Figuring Out What the Hacked Servo Does

The duty constants follow the angle convention of a normal servo, but on the hacked SG90 they set speed. **Getting a modified servo to move slowly and predictably took a few rounds.** We started with sample motor code, tuned the speed, and then added a slow-movement script. `gradual_move` smooths transitions and controls effective speed, with 10 steps by default and a 50 ms delay between them. **The class ended up shaped around what the hacked servo actually does rather than what the datasheet angles say.**

## Results

### Where the Lines Went

<figure data-figure="chart:projects/pillifyai/pillifyai-code-size"></figure>

**Most of the build went into software around a small amount of hardware code.** The app is the largest part at 5,954 lines and the server is 3,364. The hardware is small: 383 lines of Pi scripts and a 307-line ESP32 sketch. The server has 10 handlers in `auth.js`, 6 in `medications.js` and 4 in `patients.js`, plus the device routes.

### Servo Pulse Widths

<figure data-figure="chart:projects/pillifyai/pillifyai-servo-pulse"></figure>

**The three duty constants correspond to pulse widths of 0.50 ms, 1.50 ms and 2.50 ms at a 20 ms period.** Hobby servos commonly use 1.50 ms as the neutral point and about 1 to 2 ms as the typical range<sup>[[2]](#ref-2)</sup>, so the 0.50 and 2.50 ms endpoints are the extreme ends of what the code asks for. For a continuous-rotation servo, neutral means stopped.

### The Melody

**The Für Elise note list has 40 entries, 35 notes and 5 rests, and runs 9.2 seconds of nominal duration**, or 13.2 seconds counting the 0.1 second default gap after each entry.

## What I Would Change

### Dosing Logic Beyond a Frequency Enum

The schedule comes from five frequencies, from `daily` to `as_needed`. A real product would need actual dosing logic and interaction checking before a dose row is ever written. That was out of scope for one night, and it's the first thing I'd add.

### Add Push

The sketch already has a `remoteDispense` function. A path where the server triggers the dispense would remove the up-to-30-second wait, though it needs a connection the home network allows.

## References

1. <span id="ref-1"></span>Roy T. Fielding. *Architectural Styles and the Design of Network-based Software Architectures*. Doctoral dissertation, University of California, Irvine, 2000. [link](https://ics.uci.edu/~fielding/pubs/dissertation/top.htm)
2. <span id="ref-2"></span>Wikipedia. *Servo control*. [link](https://en.wikipedia.org/wiki/Servo_control)
3. <span id="ref-3"></span>Google. *Verify ID tokens*. Firebase Authentication documentation. [link](https://firebase.google.com/docs/auth/admin/verify-id-tokens)
4. <span id="ref-4"></span>Expo. *Introduction to Expo Router*. Expo documentation. [link](https://docs.expo.dev/router/introduction/)
5. <span id="ref-5"></span>Benoit Blanchon. *ArduinoJson*. Library documentation. [link](https://arduinojson.org/)
6. <span id="ref-6"></span>Espressif Systems. *Arduino core for the ESP32*. GitHub repository. [link](https://github.com/espressif/arduino-esp32)
7. <span id="ref-7"></span>Ben Croston. *RPi.GPIO*. Python Package Index. [link](https://pypi.org/project/RPi.GPIO/)
