/**
 * Olfactometer Delivery System
 * Author: Alisa Y. Hathaway, 2026
 * 
 * Controls odor delivery synchronized with breathing patterns detected via humidity sensing.
 * Delivers odor on the second inhale after an exhale is detected.
 * 
 * Key Features:
 * - Detects inhale/exhale cycles using BME280 humidity sensor
 * - Randomizes odor presentation with 15-30s inter-trial intervals
 * - Flushes system with water between trials to prevent contamination
 * - Sends TTL pulses for EEG synchronization
 * 
 */

// ============================================================================
// HARDWARE PIN ASSIGNMENTS
// ============================================================================

int solenoidPin1 = 12;  // Channel 1: water/no odor control
int solenoidPin2 = 11;  // Channel 2: odor 1
int solenoidPin3 = 10;  // Channel 3: odor 2
int solenoidPin4 = 9;   // Channel 4: odor 3
int solenoidPin5 = 8;   // Master delivery valve 

int smellPin = 400;     // Current active odor pin
int TTLpin;             // Current TTL output pin for EEG sync

// ============================================================================
// TRIAL CONFIGURATION
// ============================================================================

int num_trials = 1;

// Condition randomization - ensures balanced presentation across blocks of 8 trials
const int CONDITION_BUFFER_SIZE = 8;
int ConditionBuffer[CONDITION_BUFFER_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7};
int conditionIndex = 0;

void permute_buffer() {
  for (int i = 0; i < CONDITION_BUFFER_SIZE; i++) {
    int n = random(0, CONDITION_BUFFER_SIZE);
    int value = ConditionBuffer[n];
    ConditionBuffer[n] = ConditionBuffer[i];
    ConditionBuffer[i] = value;
  }
}

// ============================================================================
// SENSOR CONFIGURATION (BME280)
// ============================================================================

#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

#define SEALEVELPRESSURE_HPA (1013.25)
Adafruit_BME280 bme;

float temp;
float hum;
float pressure;

// ============================================================================
// SIGNAL PROCESSING BUFFERS
// ============================================================================

const int WINDOW_SIZE = 10;
float READINGS_TEMP[WINDOW_SIZE];
float READINGS_HUM[WINDOW_SIZE];
int currentIndex = 0;

// ============================================================================
// BREATHING CYCLE DETECTION
// ============================================================================

bool flag_inhale_1 = false;  // First inhale detected
bool flag_inhale_2 = false;  // Second inhale detected (triggers odor)
bool flag_exhale = false;    // Exhale detected

const int CHECK_NUM = 3;     // Number of samples to verify trend
float derivative_hum;
int newIndex;
int last_index;
bool delay_check = true;     // When true, skips breathing detection

float sampleInterval = 10.0;

/**
 * Calculates derivative between current and previous sample
 * Handles circular buffer wraparound
 */
float get_deriv(float buffer[], int index) {
  if (index == 0) {
    last_index = WINDOW_SIZE - 1;
  } else {
    last_index = index - 1;
  }
  return buffer[index] - buffer[last_index];
}

/**
 * Detects first inhale - sustained negative humidity derivative
 * Verifies trend over CHECK_NUM samples to avoid false positives
 */
void check_inhale1_deriv() {
  derivative_hum = get_deriv(READINGS_HUM, currentIndex);
  
  if (derivative_hum < 0.0) {
    for (int i = 1; i <= CHECK_NUM; i++) {
      newIndex = (currentIndex - i + WINDOW_SIZE) % WINDOW_SIZE;
      derivative_hum = get_deriv(READINGS_HUM, newIndex);
      
      if (derivative_hum > 0.0) {
        flag_inhale_1 = false;
        return;
      }
    }
    flag_inhale_1 = true;
    return;
  } else {
    flag_inhale_1 = false;
    return;
  }
}

/**
 * Detects second inhale using peak transition method
 * Pattern: positive → negative → negative (exhale peak → inhale)
 */
void check_inhale2_deriv() {
  int prevIndex = (currentIndex - 1 + WINDOW_SIZE) % WINDOW_SIZE;
  int nextIndex = (currentIndex + 1) % WINDOW_SIZE;

  float prev_deriv = get_deriv(READINGS_HUM, prevIndex);
  float curr_deriv = get_deriv(READINGS_HUM, currentIndex);
  float next_deriv = get_deriv(READINGS_HUM, nextIndex);

  if (prev_deriv >= 0.0 && curr_deriv < 0.0 && next_deriv < 0.0) {
    flag_inhale_2 = true;
  } else {
    flag_inhale_2 = false;
  }
}

/**
 * Detects exhale - sustained positive humidity derivative
 * Verifies trend over CHECK_NUM samples
 */
void check_exhale_deriv() {
  derivative_hum = get_deriv(READINGS_HUM, currentIndex);
  
  if (derivative_hum > 0.0) {
    for (int i = 1; i <= CHECK_NUM; i++) {
      newIndex = (currentIndex - i + WINDOW_SIZE) % WINDOW_SIZE;
      derivative_hum = get_deriv(READINGS_HUM, newIndex);
      
      if (derivative_hum < 0.0) {
        flag_exhale = false;
        return;
      }
    }
    flag_exhale = true;
    return;
  } else {
    flag_exhale = false;
    return;
  }
}

// ============================================================================
// MAIN SENSING AND CONTROL
// ============================================================================

/**
 * Samples sensors and monitors breathing pattern
 * State machine: waits for inhale → exhale → second inhale
 */
void check_temp(int num_times) {
  for (int i = 0; i < num_times; i++) {
    // Read and log sensor data
    temp = bme.readTemperature();
    hum = bme.readHumidity();
    pressure = bme.readPressure();
    
    Serial.print(temp); Serial.print(", ");
    Serial.print(hum); Serial.print(", ");
    Serial.print(pressure);
    Serial.println();
    
    READINGS_TEMP[currentIndex] = temp;
    READINGS_HUM[currentIndex] = hum;
    
    // Breathing detection state machine (only when delay_check is false)
    if (delay_check == false) {
      if (digitalRead(solenoidPin5) == LOW) {
        // State 1: Looking for first inhale
        if (flag_inhale_1 == false) {
          check_inhale1_deriv();
          if (flag_inhale_1 == true) {
            Serial.println("inhale!");
          }
        }
        
        // State 2: Looking for exhale after first inhale
        if (flag_inhale_1 == true) {
          check_exhale_deriv();
          if (flag_exhale == true) {
            currentIndex = (currentIndex + 1) % WINDOW_SIZE;
            return;
          }
        }
      } else {
        // State 3: Looking for second inhale (triggers odor delivery)
        check_inhale2_deriv();
        if (flag_inhale_2 == true) {
          currentIndex = (currentIndex + 1) % WINDOW_SIZE;
          return;
        }
      }
    }
    
    delay(sampleInterval);
    delay(100);
    currentIndex = (currentIndex + 1) % WINDOW_SIZE;
  }
  return;
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Configure solenoid outputs
  pinMode(solenoidPin1, OUTPUT);
  pinMode(solenoidPin2, OUTPUT);
  pinMode(solenoidPin3, OUTPUT);
  pinMode(solenoidPin4, OUTPUT);
  pinMode(solenoidPin5, OUTPUT);
  
  // Configure TTL outputs for EEG (pins 2-7)
  for (int pin = 2; pin <= 7; pin++) {
    pinMode(pin, OUTPUT);
  }
  
  // Set TTL pins HIGH (inverted logic for EEG system)
  digitalWrite(2, HIGH);
  digitalWrite(3, HIGH);
  digitalWrite(4, HIGH);
  digitalWrite(5, HIGH);
  digitalWrite(6, HIGH);
  digitalWrite(7, HIGH);

  // Initialize all solenoids to closed
  digitalWrite(solenoidPin1, LOW);
  digitalWrite(solenoidPin2, LOW);
  digitalWrite(solenoidPin3, LOW);
  digitalWrite(solenoidPin4, LOW);
  digitalWrite(solenoidPin5, LOW);

  // Initialize serial and sensor
  Serial.begin(19200);
  while(!Serial);
  Serial.println(F("BME280 start"));
  
  unsigned status;
  Wire.begin();
  status = bme.begin(0x77);
  
  if (!status) {
    Serial.println("Could not find a valid BME280 sensor, check wiring, address, sensor ID!");
    Serial.print("SensorID was: 0x");
    Serial.println(bme.sensorID(), 16);
    while (1) delay(10);
  }
  
  // Send start pulse to EEG
  digitalWrite(2, LOW);
  Serial.print("Start");
  delay(100);
  digitalWrite(2, HIGH);
  
  // Fill buffers with baseline readings
  delay_check = true;
  check_temp(20);
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Select odor for this trial using randomized block design
  int smell;
  
  if (conditionIndex == CONDITION_BUFFER_SIZE) {
    conditionIndex = 0;
  }
  if (conditionIndex == 0) {
    permute_buffer();
  }
  
  // Map buffer value to odor (2 trials per odor per block of 8)
  int val = ConditionBuffer[conditionIndex];
  if (val == 0 || val == 1) {
    smell = 1;  // Water/no odor
  } else if (val == 2 || val == 3) {
    smell = 2;
  } else if (val == 4 || val == 5) {
    smell = 3;
  } else {
    smell = 4;
  }
  conditionIndex += 1;

  // Set pins for selected odor
  switch (smell) {
    case 1:
      smellPin = solenoidPin1;
      TTLpin = 3;
      break;
    case 2:
      smellPin = solenoidPin2;
      TTLpin = 4;
      break;
    case 3:
      smellPin = solenoidPin3;
      TTLpin = 5;
      break;
    case 4:
      smellPin = solenoidPin4;
      TTLpin = 6;
      break;
    default:
      smellPin = 400;
      break;
  }
  
  // Open odor channel and fill buffers
  digitalWrite(smellPin, HIGH);
  delay_check = true;
  check_temp(10);

  // Run trial
  for (int i = 0; i < num_trials; i++) {
    delay_check = false;
    
    // Wait for first inhale (up to 10 seconds)
    check_temp(100);
    
    // After first inhale + exhale, open delivery valve
    if (flag_inhale_1 == true && flag_exhale == true && digitalRead(solenoidPin5) == LOW) {
      Serial.println("EXHALE!");
      digitalWrite(solenoidPin5, HIGH);
      
      // Wait for second inhale (up to 10 seconds)
      check_temp(100);
    }
    
    // Deliver odor on second inhale
    if (flag_inhale_2 == true && digitalRead(solenoidPin5) == HIGH) {
      unsigned long currentTime = millis();
      
      // Send TTL pulse to EEG (50ms, inverted logic)
      digitalWrite(TTLpin, LOW);
      delay(50);
      digitalWrite(TTLpin, HIGH);
      
      Serial.print("SMELL ");
      Serial.print(smell);
      Serial.print(" start: ");
      Serial.println(currentTime);
      
      // Deliver odor for 1 second
      delay_check = true;
      check_temp(10);
      
      // Close delivery valve
      digitalWrite(solenoidPin5, LOW);
    }
    
    // Reset flags for next trial
    flag_inhale_1 = false;
    flag_inhale_2 = false;
    flag_exhale = false;
  }
  
  // Close odor channel
  digitalWrite(smellPin, LOW);
  
  // Flush with water for random interval (15-30 seconds)
  int rand_delay = random(150, 300);
  digitalWrite(solenoidPin1, HIGH);
  digitalWrite(solenoidPin5, HIGH);
  Serial.println("water");
  
  delay_check = true;
  check_temp(rand_delay);
  
  digitalWrite(solenoidPin5, LOW);
  digitalWrite(solenoidPin1, LOW);
}