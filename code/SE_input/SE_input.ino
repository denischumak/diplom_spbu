#include <ADS1X15.h>


//ADS1115 ads1(0x49);
ADS1115 ads2(0x48);

const unsigned int MAX_COUNT = 10000;


unsigned int ttime = 0;
unsigned int ttime2 = 0;
unsigned int count = 0;
void setup() {
  Serial.begin(921600);
  while (!Serial);
  delay(5000);

  Wire.begin();
  Wire.setClock(400000);

  if (!ads2.begin()) {
    Serial.println("ADS not found");
    while (1) delay(100);
  }
  //   if (!ads1.begin()) {
  //   Serial.println("FIG TEBE");
  //   while (1) delay(100);
  // }

  
  // ads1.setGain(1);
  // ads1.setDataRate(6);
  ads2.setGain(1);
  ads2.setDataRate(6);

  //pinMode(A0, INPUT);
  //attachInterrupt(digitalPinToInterrupt(ALERT_PIN), alertISR, RISING);
  // unsigned long startTime = millis();
  // while (millis() - startTime < 90000) {
  //   Serial.println("w");
  //   delay(1000);
  // }
  // ttime = micros();
}


// 60HZ, 1.65
void loop() {

  // if (micros() - ttime >= 10'000) {
  //   int16_t raw = analogRead(A0);
  //   Serial.println(raw);
  //   ++count;
  //   ttime = micros();
  
  // Serial.print(ads2.readADC(0) * 0.000125);
  // Serial.print(" ");
  Serial.print(ads2.readADC(0));
  Serial.print(" ");
  Serial.print(ads2.readADC(1));
  Serial.print(" ");
  Serial.print(ads2.readADC(2));
  Serial.print(" ");
  Serial.print(ads2.readADC(3));

  // Serial.print(" ");
  // Serial.print(ads1.readADC(0));
  // Serial.print(" ");
  // Serial.print(ads1.readADC(1));
  // Serial.print(" ");
  // Serial.print(ads1.readADC(2));
  // Serial.print(" ");
  // Serial.print(ads1.readADC(3));
  Serial.print("\n");
  // Serial.print(ads1.readADC(3));
  // Serial.print("\n");
  delay(10);
  // if (count == MAX_COUNT) {
  //   while(1);
  // }

}



// #include <ADS1X15.h>


// ADS1115 ads1(0x49);
// ADS1115 ads2(0x48);

// const unsigned int MAX_COUNT = 10000;


// unsigned int ttime = 0;
// unsigned int ttime2 = 0;
// unsigned int count = 0;
// void setup() {
//   Serial.begin(921600);
//   while (!Serial);
//   //delay(5000);

//   Wire.begin();
//   Wire.setClock(400000);

//   if (!ads2.begin()) {
//     Serial.println("ADS not found");
//     while (1) delay(100);
//   }
//   if (!ads1.begin()) {
//     Serial.println("FIG TEBE");
//     while (1) delay(100);
//   }

  
//   ads1.setGain(1);
//   ads1.setDataRate(6);
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
//   pinMode(A2, INPUT);
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
//   Serial.println(analogRead(A2) * 0.000806);
  

//}