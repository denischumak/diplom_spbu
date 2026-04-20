#include <ADS1X15.h>

#define ALERT_PIN_ADS3 0
#define ALERT_PIN_ADS2 1
#define ADS3_ADDR 0x48
#define ADS2_ADDR 0x49
#define ADS_GAIN 8
#define ADS_DATA_RATE 7 // 860SPS

ADS1115 ads3(ADS3_ADDR); // single-shot mode by default
ADS1115 ads2(ADS2_ADDR); // single-shot mode by default

volatile bool new_data_ready_ads3 = false;
volatile bool new_data_ready_ads2 = false;

void IRAM_ATTR on_alert_ADS3() {
  new_data_ready_ads3 = true;
}
void IRAM_ATTR on_alert_ADS2() {
  new_data_ready_ads2 = true;
}

//const unsigned int MAX_COUNT = 10000;
unsigned long ttime1 = 0;

void setup() {
  Serial.begin(921600);
  while (!Serial);
  delay(10);

  Wire.begin();
  Wire.setClock(400000);

  if (!ads3.begin()) {
    Serial.println("ADS1 not found");
    while (1) delay(100);
  }
  if (!ads2.begin()) {
    Serial.println("ADS not found");
    while (1) delay(100);
  }
  
  ads3.setGain(ADS_GAIN);
  ads3.setDataRate(ADS_DATA_RATE);
  ads3.setComparatorThresholdLow(0x0000);
  ads3.setComparatorThresholdHigh(0x8000);
  ads3.setComparatorQueConvert(0);
  pinMode(ALERT_PIN_ADS3, INPUT);
  attachInterrupt(digitalPinToInterrupt(ALERT_PIN_ADS3), on_alert_ADS3, RISING); // COMP_POL = 1 by default

  ads2.setGain(ADS_GAIN);
  ads2.setDataRate(ADS_DATA_RATE);
  ads2.setComparatorThresholdLow(0x0000);
  ads2.setComparatorThresholdHigh(0x8000);
  ads2.setComparatorQueConvert(0);
  pinMode(ALERT_PIN_ADS2, INPUT);
  attachInterrupt(digitalPinToInterrupt(ALERT_PIN_ADS2), on_alert_ADS2, RISING); // COMP_POL = 1 by default

  // ttime1 = micros();
  // ads2.requestADC_Differential_1_3();
}

unsigned long avg_time = 0;
int SAMPLES_MAX = 10000;
int samples = 0;
unsigned int max_val = 0;
unsigned int min_val = 65'535;

void loop() { //860sps полный цикл чтения 1.5мс, время на запрос - 140мкс (макс 147), чтение - 210 мкс (макс 255), пин2 ads3 - большой палец 
  
  ads3.requestADC_Differential_0_3();

  while (!new_data_ready_ads3);
  new_data_ready_ads3 = false;
  ttime1 = micros();
  int16_t val = ads3.getValue();
  unsigned long cur_time = (unsigned long)micros() - ttime1;
  avg_time += cur_time;
  ++samples;
  if (cur_time > 5000) {
    Serial.println("SAMPLE NUMBER: ");
    Serial.println(samples);
  }
  max_val = cur_time > max_val ? cur_time : max_val;
  min_val = cur_time < min_val ? cur_time : min_val;

  // if (ads3.isReady()) {
  //   int16_t val = ads3.getValue();
  //   unsigned long cur_time = (unsigned long)micros() - ttime1;
  //   avg_time += cur_time;
  //   ++samples;
  //   if (cur_time > 5000) {
  //     Serial.println("SAMPLE NUMBER: ");
  //     Serial.println(samples);
  //   }
  //   max_val = cur_time > max_val ? cur_time : max_val;
  //   min_val = cur_time < min_val ? cur_time : min_val;
  //   ttime1 = micros();
  //   ads3.requestADC_Differential_0_3();
  // }
  if (samples == SAMPLES_MAX) {
    Serial.println(avg_time / SAMPLES_MAX);
    Serial.println(max_val);
    Serial.println(min_val);

    while(1);
  }

}