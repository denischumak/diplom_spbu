#include <Wire.h>
#include <BMI160.hpp>
#include <MadgwickAHRS.h>

BMI160 IMU;
Madgwick filter;
const unsigned int FREQ = 100;
const unsigned long delayTime = 1'000'000 / FREQ;
unsigned long microsPrev;

calData calibration = {
  true,
  {-0.023660, -0.102030, 0.001446}, // accel Bias
  {-0.466675, -0.429138, 0.964844}, // gyro Bias
  {1.001412, 0.991496,  1.001503} // accel Scale
};
AccelData accel;
GyroData gyro;

void setup() {
  Serial.begin(921600);
  while (!Serial);
  Wire.begin();
  Wire.setClock(400000);
  
  int err = IMU.init(calibration); // Адрес 0x69

  IMU.setGyroRange(1000);
  IMU.setAccelRange(4);
  if (err != 0) {
    Serial.print("Ошибка инициализации BMI160, код: ");
    Serial.println(err);
    while (1);
  }
  filter.begin(FREQ);
  Serial.println("BMI160 готов к работе!");
  microsPrev = micros();
}

void loop() {
  float roll, pitch, yaw;
  unsigned long microsNow;
  microsNow = micros();
    if (microsNow - microsPrev >= delayTime) {
    IMU.updateGyroAccel();
    IMU.getAccel(&accel);
    IMU.getGyro(&gyro);
    filter.updateIMU(gyro.gyroX, gyro.gyroY, gyro.gyroZ, accel.accelX, accel.accelY, accel.accelZ);
    roll = filter.getRoll();
    pitch = filter.getPitch();


    Serial.print(roll);
    Serial.print(" ");
    Serial.print(pitch);
    Serial.print(" ");
    Serial.println(gyro.gyroZ);
    microsPrev = micros();
  }
  // IMU.updateGyroAccel();
  // IMU.getAccel(&accel);
  // IMU.getGyro(&gyro);
  // Serial.print(gyro.gyroX);
  // Serial.print(" ");
  // Serial.print(gyro.gyroY);
  // Serial.print(" ");
  // Serial.println(gyro.gyroZ);
  // delay(50);
}