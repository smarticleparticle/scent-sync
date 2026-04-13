# Respiration-Synchronized Olfactometer - Build Instructions

## Overview

This low-cost olfactometer delivers odors synchronized with natural nasal breathing for olfactory research. The system delivers scented air at ~3 L/min to prevent nasal discomfort and uses real-time humidity sensing to detect respiratory phase and trigger odor delivery at precise moments in the breathing cycle.

**Key Features:**
- Delivers 3 distinct odorants plus water control
- Respiration-synchronized delivery using BME280 sensor
- Arduino-controlled with TTL pulse output for EEG synchronization
- Directs unused scented air outside experimental room
- Battery-powered solenoids to eliminate 60Hz electrical noise

---

## System Components

### A. Air Delivery System

**Air Compressor:** Spriek 'Intelligent Digital Display Nebulizer for Adults and Kids'
- Noise level: ≤50 dB
- Provides controlled airflow at 3 L/min
- **Note:** Various air compressor nebulizers may work if they have universal connection tubing and sufficient air supply. Avoid flow rates >5 L/min as this can cause nasal drying and irritation without temperature/humidity control.

### B. Airflow Distribution

- **Flow Control:** Flowmeters (0-5 L/min range) - one per odor channel
- **Tubing Connection:** "Y" connectors arranged in binary tree structure
- **Tubing:** Tygon tubing
  - Outlet: 7.94mm ID x 12.7mm OD (5/16" ID x 1/2" OD)
  - Inlet: 9.53mm ID x 15.88mm OD (3/8" ID x 5/8" OD)

### C. Odor Containers (Bubble Humidifiers)

- **Container:** Ball® wide-mouth mason jars (glass)
  - Glass chosen for nonporous material, affordability, ease of cleaning.
- **Safety:** Pop-off pressure valves rated for 1-5 PSI
- **Housing:** 3D printed holder
  - Accommodates odor containers in groups of 2
  - Includes space between top and bottom for flowmeter and tubing placement

**Assembly:**
1. Drill precise holes in jar lids for tubing and pressure valve using metal laser cutter
2. Install inlet and outlet tubing through drilled holes
3. Install pressure relief valve

**Note on bubble humidifiers:** Lower flow rates (<10 L/min) allow gas to effectively pass through water from bottom to surface, resulting in increased water vapor. Flow rate of 0-5 L/min is used for this reason.

**Note on temperature control:** This design does not include heated air. While heated humidifiers increase moisture, increased heat allows for more bacteria colony clustering.

### D. Valve System

**Primary Valves:** Four 3-way solenoid valves (one per odor channel)
- Ports 1 & 2 connected when unpowered (directs air outside room)
- Ports 1 & 3 connected when powered (directs air toward subject)

**Secondary Valve:** One 3-way solenoid valve before nasal cannula
- Controls final delivery timing
- Directs excess air outside room when not delivering odor

**Protective Components:** Diodes connected in parallel to each solenoid to protect switching transistor
- Provides unidirectional path for current when solenoid deactivates

### E. Delivery Interface

**Nasal Cannula:** Standard medical nasal cannula
- Inner tube diameter: 2.5mm
- Length: 1.21 meters (4 feet)
- Can be purchased with nonstick materials like Teflon
- Allows bilateral nostril delivery without active sniffing
- Total travel time for scented air through cannula: ~1 second at flow rate of ~0.35 L/min

### F. Respiratory Monitoring

- **Sensor:** Bosch BME280 temperature/pressure/humidity sensor
- **Mounting:** Small flexible PCB fabricated with BME280
- **Placement:** Inserted into center of nasal cannula between the two nostrils
- **Connection:** Level shifter required for BME280 connection to Arduino (see schematic)

### G. Control Electronics

**Microcontroller:** Arduino Uno
- digitalRead() and digitalWrite() latency: ~3.4 µs average

**Power Supply:** External 2500mAh lithium-ion battery for solenoids
- Battery power eliminates 60Hz electrical noise in EEG recordings

**TTL Output:** Digital pins for EEG synchronization
- Inverted logic (LOW = pulse, HIGH = idle)
- 50ms pulse duration

### H. Additional Components

- **Tubing:** Routes excess air outside experimental room
- **Rollable Cart:** For transportation between lab and experimental room

---

## Assembly Instructions

### 1. Construct Odor Containers

1. Use metal laser cutter to drill precise holes in Ball® jar lids for:
   - Inlet tubing (9.53mm ID)
   - Outlet tubing (7.94mm ID)
   - Pressure relief valve
2. Install tubing through holes
3. Install 1-5 PSI pressure relief valves
4. Place jars in 3D printed housing

### 2. Assemble Airflow System

1. Connect air compressor nebulizer output to binary tree of "Y" connectors
2. Install flowmeter on each branch (4 total)
3. Connect flowmeter outputs to odor container inlets
4. Connect odor container outlets to 3-way solenoid valves (primary valves)
5. Route unused air ports to outside experimental room
6. Merge all four scent lines after solenoids
7. Install final 3-way solenoid valve before nasal cannula
8. Route excess air to outside experimental room
9. Connect final output to nasal cannula

### 3. Fabricate Respiratory Sensor

1. Create small flexible PCB with Bosch BME280 sensor
2. Insert sensor assembly into center of nasal cannula between nostril ports

### 4. Build Electronics

1. Assemble circuit according to schematic:
   - Connect solenoids to Arduino digital outputs through transistor switches
   - Install protective diode in parallel with each solenoid
   - Connect external lithium-ion battery to power solenoids
   - Wire BME280 to Arduino through level shifter
   - Configure TTL output pins for EEG system
2. Upload Arduino code (see software repository)

### 5. System Integration

1. Mount all components on rollable cart
2. Verify all connections
3. Route exhaust tubing outside experimental room

---

## Operating Protocol

The system operates in four synchronized stages:

### Stage 1: Breath Detection

- Arduino continuously monitors BME280 humidity measurements
- Calculates derivatives within rolling window to detect respiratory phase transitions
- **Inhalation onset:** Derivatives transition from positive to negative at humidity peak, followed by sustained negative derivatives
- **Exhalation onset:** Sustained positive derivatives indicate increasing humidity from exhaled air

### Stage 2: Odor Preparation

- Random inter-stimulus interval: 15-30 seconds
- Activates selected odor channel's primary solenoid for up to 10 seconds
- Draws scented air through bubble humidifier

### Stage 3: Timed Release

- Upon detection of exhalation (following first inhalation), secondary solenoid opens for 1 second
- Scented air traverses 1.21m nasal cannula
- Timed to arrive at nose at onset of subsequent inhalation

### Stage 4: EEG Synchronization

- TTL pulse transmitted to EEG system when second inhalation is detected
- Marks precise timestamp of inhalation onset (scent arrival)
- Accounts for minimal electrical delays (microseconds)

---

## Flow Rate Calculations

**After secondary solenoid:**
- Average flow rate: Q ≈ 0.35 L/min ≈ 5.83×10⁻⁶ m³/s
- Nasal cannula inner diameter: 2.5mm
- Cross-sectional area: A = π(0.00125)² = 4.91×10⁻⁶ m²
- Velocity: v ≈ Q/A = 1.19 m/s
- Travel time through 1.2m cannula: t ≈ 1.2m / 1.19m/s = 1.00s

This ensures scented air arrives during the subject's inhalation.

---

## Safety Considerations

1. **Pressure Relief:** 1-5 PSI pop-off valves protect against over-pressurization
2. **Flow Rate Limits:** Maximum 3 L/min to nasal cannula to prevent nasal discomfort
3. **Material Selection:** Glass containers (non-porous), Tygon tubing, Teflon-compatible nasal cannulas
4. **Electrical Isolation:** Battery-powered solenoids eliminate electrical noise and shock hazards
5. **Hygiene:** Disposable nasal cannulas, easily cleanable glass containers

---

## Bill of Materials

### Major Components

- Spriek Intelligent Digital Display Nebulizer
- Arduino Uno
- Bosch BME280 sensor
- 5× 3-way solenoid valves
- 4× Flowmeters (0-5 L/min)
- 4× Ball® wide-mouth mason jars
- 4× Pressure relief valves (1-5 PSI)
- 2500mAh lithium-ion battery
- Medical nasal cannula
- Protective diodes (1 per solenoid)

### Consumables

- Tygon tubing (two sizes)
- "Y" connectors
- Essential oils (peppermint, oregano, grapefruit)
- Deionized water
- Flexible PCB material
- 3D printing filament for housing

### Electronics Components

- Transistors for solenoid switching
- Level shifter for BME280
- Wire and connectors
- TTL output connectors

---

## References

Design considerations based on:
- Flow rate recommendations for nasal comfort
- Bubble humidifier performance characteristics
- Sensor placement for optimal respiratory monitoring
- TTL pulse timing for EEG synchronization

For Arduino code and additional documentation, see the project repository.
