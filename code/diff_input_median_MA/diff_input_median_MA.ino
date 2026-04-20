#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;


const float   alpha = 0.5f;          // размер окна скользящего среднего
const int   CAL_SAMPLES = 384;         // сколько выборок брать для калибровки 
const float GAIN_8_MV_CONST = 0.015625f;
const uint8_t ADDR = 0x48;

// Смещение нуля (в сырых отсчётах АЦП)
float offset_raw = 0;

// Калибровка нуля
void calibrateZero() {
  Serial.println("Calibrating zero...");
  delay(2000);
  long sum = 0;
  for (int i = 0; i < CAL_SAMPLES; i++) {
    sum += ads.getLastConversionResults();
  }
  offset_raw = sum / (float)CAL_SAMPLES;
}



float readRawDiff() {
  return (float)ads.getLastConversionResults() - offset_raw;  
}

unsigned int ttime = 0;
bool is_new_value = true;
float current;


void setup() {
  Serial.begin(115200);
  while (!Serial) { 
    delay(10); 
  }
  
  if (!ads.begin(ADDR)) {
    Serial.println("Ошибка! ADS1115 не найден.");
    while (1);
  }
  
  ads.setGain(GAIN_EIGHT);       // ±0.512 В
  ads.setDataRate(RATE_ADS1115_250SPS);
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_1_3, /*continuous=*/true);
  calibrateZero();
}

void inline EMA(float& current, float new_val) {
  if (is_new_value) {
    is_new_value = false;
    current = new_val;
    return;
  }
  current += (new_val - current) * alpha;
}

int MAXCNT = 10000;
int count = 0; 

void loop() {
  float new_val = readRawDiff();
  EMA(current, new_val);
  ++count;
  if (count > MAXCNT) {
    while (1);
  }
  Serial.println(current * GAIN_8_MV_CONST); // mV
  delayMicroseconds(16015);
}








