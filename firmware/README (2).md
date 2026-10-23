# Firmware

`olfactometer-sync-2026.ino` is the Arduino sketch that runs the respiration-synchronized olfactometer. It reads nasal humidity from a BME280 sensor, detects the breathing phase, opens the odor valves so that the odor reaches the nose at the start of an inhalation, and sends a TTL pulse to the EEG system at that moment. For the hardware, see [`../hardware/`](../hardware/); for assembly and the operating protocol, see [`../documentation/build-guide.md`](../documentation/build-guide.md).

## Requirements

| item | value |
|---|---|
| Board | Arduino Uno |
| Libraries | Adafruit BME280 Library, Adafruit Unified Sensor (`Wire` and `SPI` ship with the Arduino IDE) |
| Library versions used | `[FILL IN: e.g. Adafruit BME280 Library x.y.z, Adafruit Unified Sensor x.y.z]` |
| Arduino IDE / core version | `[FILL IN]` |
| Sensor | BME280 on I2C, address `0x77`, connected through the 3.3 V / 5 V level converter (see the schematic) |
| Serial | 19200 baud |

## Upload

1. Install the two libraries from the Arduino Library Manager.
2. Open `olfactometer-sync-2026.ino`, select **Arduino Uno** and the correct port, and upload.
3. Open the Serial Monitor at **19200 baud**. You should see `BME280 start`, and then a stream of sensor readings.

If the sketch prints `Could not find a valid BME280 sensor, check wiring, address, sensor ID!` and stops, check the I2C wiring, the level converter, and the address (`0x77` here; many breakout boards use `0x76`).

## Pin map

| Arduino pin | Function | Notes |
|---|---|---|
| 12 | Channel 1 valve: water (no odor) | |
| 11 | Channel 2 valve: odor 1 | |
| 10 | Channel 3 valve: odor 2 | |
| 9 | Channel 4 valve: odor 3 | |
| 8 | Delivery valve (before the nasal cannula) | Open = air goes toward the subject |
| 2 | TTL: session start | One 100 ms pulse after the sensor is found |
| 3 | TTL: water trial | One 50 ms pulse per delivery |
| 4 | TTL: odor 1 trial | One 50 ms pulse per delivery |
| 5 | TTL: odor 2 trial | One 50 ms pulse per delivery |
| 6 | TTL: odor 3 trial | One 50 ms pulse per delivery |
| 7 | TTL: unused | Configured as an output, idle HIGH |
| A4 / A5 | I2C SDA / SCL to the BME280 | Through the level converter |

The valve pins drive the TIP120 transistors shown in the schematic (`../hardware/schematics/olfactometer_schematic.pdf`). **TTL logic is inverted:** the pins idle HIGH and pulse LOW.

## What the sketch does

On start-up the sketch sets all valves closed and the TTL pins HIGH, finds the sensor, sends the start pulse on pin 2, and reads 20 baseline samples.

It then repeats the following trial indefinitely:

1. **Choose the condition.** Conditions come in blocks of 8 trials, shuffled within each block, with 2 trials each for water, odor 1, odor 2 and odor 3.
2. **Open the channel.** The chosen channel valve opens and 10 samples are collected (about 1 s).
3. **Wait for a breath.** The sketch waits up to 100 samples (about 11 s) for a falling-humidity phase (flagged as inhalation), followed by a rising-humidity phase (exhalation). A phase is only accepted if the humidity did not move the other way during the previous 3 samples.
4. **Open the delivery valve** (pin 8) when the exhalation is detected.
5. **Wait for the next inhalation**, for up to 100 samples (about 11 s): a humidity peak followed by falling humidity.
6. **Deliver.** When that inhalation is detected, the sketch sends a 50 ms TTL pulse on the channel's TTL pin, keeps the delivery valve open for about 1 s, and closes it.
7. **Flush.** The channel closes, and water flows through the channel 1 and delivery valves for a random 150 to 299 samples (about 16 to 33 s), after which both close and the next trial starts.

Every sample is also printed to the serial port as `temperature (°C), humidity (%RH), pressure (Pa)`, followed by status messages such as `inhale!`, `EXHALE!`, `SMELL <channel> start: <ms>` and `water`. The `Start` message is printed without a line break.

## Checking that it works

1. With the sensor connected, breathe on it. The humidity column should rise during exhalation and fall during inhalation, and `inhale!` and `EXHALE!` should appear at the corresponding phases.
2. With the valves connected, you should hear them switch, and a `SMELL <n> start: <ms>` line should appear for each delivery.
3. With the EEG system (or an oscilloscope or logic analyzer) on the TTL pins, you should see a 50 ms LOW pulse on the channel's pin at each `SMELL` line. `[FILL IN: expected number of deliveries per block of 8, and the typical fraction of missed trials, if known]`

## Behavior to be aware of

- **Sampling rate.** Each sample includes `delay(10)` and `delay(100)`, plus the sensor read time, so the rate is a little under 9 Hz. The times above are approximate. `sampleInterval = 10.0` is only the first of the two delays. Measure the actual rate on your board if timing matters.
- **Random numbers are not seeded.** The sketch does not call `randomSeed()`, so the Arduino's `random()` produces the same sequence after every reset. Both the shuffle of conditions within each block of 8 (`permute_buffer()`) and the flush durations therefore repeat identically in every session. Each block still contains every condition exactly twice. To get a different sequence per session, add `randomSeed(analogRead(A0));` (with A0 unconnected) at the start of `setup()`.
- **Missed trials are silent.** If no breath is detected within the waiting window, no TTL pulse is sent and no message is printed, but the trial still counts toward the block of 8.
- **Intervals between deliveries** include the waiting time for breath detection, so they are longer than the 15 to 30 s random flush alone.
- **Second-inhalation check.** The detection of the second inhalation (`check_inhale2_deriv`) compares the current sample with the previous ones in a circular buffer of 10 samples. The "next" difference it uses is taken from the oldest sample in the buffer. This is the code that was used to collect the data in the paper, and it is left unchanged.

## License

See the top-level `LICENSE`.
