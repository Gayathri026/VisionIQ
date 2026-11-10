# ISO 13849-1: Safety of Machinery — Safety-Related Parts of Control Systems

## Section 1: Interlock Architecture and Stoppage Root Causes
Automated industrial machinery (stamping presses, hydraulic presses, CNC machining centers) equipped with safety controllers (PLCs conforming to Performance Level PLd or PLe) halt operations due to predefined safety functions:
- Guard Interlock Gate Actuation: Physical barrier opened during dangerous cycle.
- Electro-Sensitive Protective Equipment (ESPE / Light Curtains): Optical beam interruption in front of die hazard envelope.
- Emergency Stop (E-Stop) Actuation per ISO 13850: Category 0 or Category 1 mechanical latching stop.
- Hydraulic Pressure Drop: System accumulator or supply pressure collapsing below the minimum safety threshold (typically 120 bar).
- Thermal Motor Overload Trip: Winding temperature sensor exceeding Class F limits.

## Section 2: Troubleshooting and Diagnostic Procedures
When investigating unexpected production machinery stoppage:
1. Interrogate safety controller fault log buffer for active fault diagnostic codes (F-codes).
2. Measure optical alignment and dirt deposition on light curtain receiver optics.
3. Inspect hydraulic fluid level, valve coil resistance, and filter differential pressure gauges.
4. Verify mechanical interlock switch cam alignment and positive opening contact operation before clearing lockout/tagout (LOTO).
