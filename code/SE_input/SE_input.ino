// #include <ADS1X15.h>


// //ADS1115 ads1(0x49);
// ADS1115 ads2(0x48);

// const unsigned int MAX_COUNT = 10000;


// unsigned int ttime = 0;
// unsigned int ttime2 = 0;
// unsigned int count = 0;
// void setup() {
//   Serial.begin(921600);
//   while (!Serial);
//   delay(500);

//   Wire.begin();
//   Wire.setClock(400000);

//   if (!ads2.begin()) {
//     Serial.println("ADS not found");
//     while (1) delay(100);
//   }
//   //   if (!ads1.begin()) {
//   //   Serial.println("FIG TEBE");
//   //   while (1) delay(100);
//   // }

  
//   // ads1.setGain(1);
//   // ads1.setDataRate(6);
//   ads2.setGain(1);
//   ads2.setDataRate(6);

//   //pinMode(A0, INPUT);
//   //attachInterrupt(digitalPinToInterrupt(ALERT_PIN), alertISR, RISING);
//   // unsigned long startTime = millis();
//   // while (millis() - startTime < 90000) {
//   //   Serial.println("w");
//   //   delay(1000);
//   // }
//   // ttime = micros();
// }


// // 60HZ, 1.65
// void loop() {

//   // if (micros() - ttime >= 10'000) {
//   //   int16_t raw = analogRead(A0);
//   //   Serial.println(raw);
//   //   ++count;
//   //   ttime = micros();
  
//   // Serial.print(ads2.readADC(0) * 0.000125);
//   // Serial.print(" ");
//   Serial.print(ads2.readADC(0));
//   Serial.print(" ");
//   Serial.print(ads2.readADC(1));
//   Serial.print(" ");
//   Serial.print(ads2.readADC(2));
//   Serial.print(" ");
//   Serial.print(ads2.readADC(3));

//   // Serial.print(" ");
//   // Serial.print(ads1.readADC(0));
//   // Serial.print(" ");
//   // Serial.print(ads1.readADC(1));
//   // Serial.print(" ");
//   // Serial.print(ads1.readADC(2));
//   // Serial.print(" ");
//   // Serial.print(ads1.readADC(3));
//   Serial.print("\n");
//   // Serial.print(ads1.readADC(3));
//   // Serial.print("\n");
//   delay(10);
//   // if (count == MAX_COUNT) {
//   //   while(1);
//   // }

// }

#include <ADS1X15.h>
#include <BMI160.hpp>
#include <MadgwickAHRS.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/event_groups.h>

#define ALERT_PIN_ADS3 1
#define ADS3_ADDR 0x48
#define IMU_ADDR 0x69
#define IMU_GYRO_RANGE 1000
#define IMU_ACCEL_RANGE 4
#define ADS_GAIN 8
#define LSB_MV 0.015625
#define ADS_DATA_RATE 6   // 475 SPS
#define FREQ 100
#define WARM_UP_DELAY_MILLIS 1'000
#define FINGER_NUMBER 3
//std:   0.498702924
// p2p:      2.692733333

#define ADS3_READY_BIT (1 << 0)

ADS1115 ads3(ADS3_ADDR);

BMI160 IMU;
Madgwick filter;

calData calibration = {
  true,
  {-0.023660f, -0.102030f, 0.001446f},
  {0.464141763f, -0.42757334f, 0.957810755f},
  {1.001412f, 0.991496f, 1.001503f}
};

AccelData accel;
GyroData gyro;

struct DataPacket {
  float fingers[FINGER_NUMBER];
  AccelData accel;
  GyroData gyro;
  float quats[4];
};

QueueHandle_t dataQueue;
EventGroupHandle_t adsEvents;

void IRAM_ATTR on_alert_ADS3() {
  BaseType_t hpTaskWoken = pdFALSE;
  xEventGroupSetBitsFromISR(adsEvents, ADS3_READY_BIT, &hpTaskWoken);
  if (hpTaskWoken) {
    portYIELD_FROM_ISR();
  }
}

static inline void wait_for_ads3() {
  xEventGroupWaitBits(
    adsEvents,
    ADS3_READY_BIT,
    pdTRUE,
    pdTRUE,
    portMAX_DELAY
  );
}

void warmUpAllHallSensors() {
  Serial.println("Waiting for the equipment to warm up...");
  unsigned long start_time = millis();
  unsigned long current_time = millis();
  while (current_time - start_time < WARM_UP_DELAY_MILLIS) {
    Serial.print((WARM_UP_DELAY_MILLIS - current_time + start_time) / 1000);
    Serial.println("s left...");
    delay(1000);
    current_time = millis();
  }
  Serial.println("Warm up ended.");
}

void taskDataCollection(void *pvParameters) {
  const TickType_t period = pdMS_TO_TICKS(1000 / FREQ);
  TickType_t lastWakeTime = xTaskGetTickCount();
  DataPacket pkt;
  for (;;) {
    vTaskDelayUntil(&lastWakeTime, period);
    ads3.requestADC_Differential_1_3();  // мизинец
    wait_for_ads3();
    pkt.fingers[2] = ads3.getValue() * LSB_MV;

    ads3.requestADC_Differential_0_3();  // указательный

    IMU.updateGyroAccel();
    IMU.getAccel(&pkt.accel);
    IMU.getGyro(&pkt.gyro);
    filter.updateIMU(pkt.gyro.gyroX, pkt.gyro.gyroY, pkt.gyro.gyroZ,
                     pkt.accel.accelX, pkt.accel.accelY, pkt.accel.accelZ);

    wait_for_ads3();
    pkt.fingers[0] = ads3.getValue() * LSB_MV;

    ads3.requestADC_Differential_2_3();  // безымянный
    wait_for_ads3();
    pkt.fingers[1] = ads3.getValue() * LSB_MV;

    filter.getQuat(pkt.quats);
    xQueueOverwrite(dataQueue, &pkt);
  }
}

void taskSerialOutput(void *pvParameters) {
  DataPacket pkt;
  char line[160];
  long cnt = 0;
  long t_start, t_end;
  for (;;) {
    if (xQueueReceive(dataQueue, &pkt, portMAX_DELAY) == pdTRUE) {
      t_start = micros();
      int n = snprintf(
        line, sizeof(line),
        "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
        pkt.fingers[0], pkt.fingers[1], pkt.fingers[2], // (8b + 1b) * 3 = 27b
        pkt.accel.accelX, pkt.accel.accelY, pkt.accel.accelZ, // (6b + 1b) * 3 = 21b
        pkt.gyro.gyroX, pkt.gyro.gyroY, pkt.gyro.gyroZ, // (9b + 1b) * 3 = 30b
        pkt.quats[0], pkt.quats[1], pkt.quats[2], pkt.quats[3]
      );

      if (n > 0) {
        Serial.write((const uint8_t*)line, (size_t)n);
        Serial.println(micros() - t_start);
        ++cnt;
        if (cnt == 10'000){
          while(true){delay(100);}
        }
      }
    }
  }
}

void setup() {
  Serial.setRxBufferSize(1024); 
  Serial.begin(921600);
  while (!Serial) {
    vTaskDelay(1);
  }

  Wire.begin();
  Wire.setClock(400000);

  if (!ads3.begin()) {
    Serial.println("ADS3 not found");
    for (;;) {}
  }

  ads3.setGain(ADS_GAIN);
  ads3.setDataRate(ADS_DATA_RATE);
  ads3.setComparatorThresholdLow(0x0000);
  ads3.setComparatorThresholdHigh(0x8000);
  ads3.setComparatorQueConvert(0);
  pinMode(ALERT_PIN_ADS3, INPUT);
  attachInterrupt(digitalPinToInterrupt(ALERT_PIN_ADS3), on_alert_ADS3, RISING);

  int err = IMU.init(calibration, IMU_ADDR);
  if (err != 0) {
    Serial.print("BMI160 initialization error, code: ");
    Serial.println(err);
    for (;;) {}
  }

  IMU.setGyroRange(IMU_GYRO_RANGE);
  IMU.setAccelRange(IMU_ACCEL_RANGE);
  filter.begin(FREQ);

  adsEvents = xEventGroupCreate();
  dataQueue = xQueueCreate(1, sizeof(DataPacket));

  if (adsEvents == nullptr || dataQueue == nullptr) {
    Serial.println("RTOS objects creation failed");
    for (;;) {}
  }

  warmUpAllHallSensors();

  xTaskCreatePinnedToCore(taskDataCollection, "DataCollect", 4096, nullptr, 3, nullptr, 0);
  xTaskCreatePinnedToCore(taskSerialOutput,   "SerialOut",   4096, nullptr, 1, nullptr, 0);

  vTaskDelete(nullptr);

  
}

void loop() {
}
