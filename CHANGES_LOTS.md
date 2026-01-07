Changes for multi-lot spawning and tap control

Overview
- Add per-grain lot IDs, timed spawning from a point, and a bottom tap that opens at a specified time.
- Track discharged grains per lot and stop once a target discharge count is reached.

Files changed and why
- estructuras.h
  - Added `lot_id` to `grain_prop` so each grain carries its lot tag.
  - Added `tapOpen`, `tapOpenTime`, and `dischargeTarget` to `parameters` so kernels can enable/disable the floor.
- get_contacts.cu
  - Re-enabled the floor contact (bb = ngrains + 8) and gated it by `!pars.tapOpen` so the tap blocks the orifice until opened.
- get_forces.cu
  - Re-enabled the floor contact force, using plane parameters and `!pars.tapOpen`.
- siloTolva.cu
  - Added parsing for `lots.data` (tap open time, discharge target, and up to 3 lots).
  - Replaced the hexagonal initial fill with timed activation from per-grain spawn points.
  - Added per-lot discharge counters and early stop when `dischargeTarget` is reached.
  - Wrote `lot_discharge.dat` with per-lot and total discharged counts.
- lots.data
  - New input file that defines the tap open time, discharge target, and lot schedules.

Input format: lots.data
First non-comment line:
  tap_open_time(s) discharge_target(grains)
Then one line per lot (up to 3):
  lot_id count spawn_time(s) x y z

Example:
  0.50 200
  1 500 0.00 10.0 0.0 55.0
  2 300 0.20 10.0 0.0 55.0
  3 200 0.40 10.0 0.0 55.0

Notes
- The sum of lot counts must match `ngrains` from `siloTolva.data`.
- To make the silo square in top view, set `siloThick = siloWidth` in `siloTolva.data`.
- Discharge is counted when an active grain crosses below z < 0 after the tap is open.
