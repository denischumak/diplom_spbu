#include <Wire.h>
#include <BMI160.hpp> // Убедитесь, что библиотека установлена

// === Настройки ===
const int numReadings = 1000;   // Количество замеров для усреднения в одном положении
calData calibration;

// === Глобальные переменные ===
BMI160 IMU; // Создаём объект IMU
AccelData accel;   // Массив для хранения сырых данных (x, y, z)
GyroData gyro;

// === Прототипы функций ===
void initSensor();
void waitForUser(const char* message);
void readAverageAccel(float* avg, int samples);
void readAverageGyro(float* avg, int samples);
void calibrateAxisAccel(int axisPos, int axisNeg, float& offset, float& scale);
void printCalibrationResultAccel(const char* axisName, float offset, float scale);
void calibrateAccelerometer();
void calibrateGyroscope();

// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial);
  Wire.begin();
  Wire.setClock(400000);


  initSensor(); // 1. Инициализация датчика
  calibrateAccelerometer();
  //calibrateGyroscope();
  
  Serial.println(F("\nКалибровка завершена! Вы можете перезагрузить плату или использовать эти значения."));
}

void calibrateAccelerometer() {
  Serial.println(F("\nBMI160 КАЛИБРОВКА АКСЕЛЕРОМЕТРА (6 положений)"));
  Serial.println(F("==============================================="));
// 2. Переменные для хранения результатов
  float offsetX, offsetY, offsetZ;
  float scaleX, scaleY, scaleZ;

  // 3. Калибровка оси X
  Serial.println(F("\n--- КАЛИБРОВКА ОСИ X ---"));
  waitForUser("Положите датчик так, чтобы ось X была направлена ВВЕРХ. Нажмите Enter...");
  calibrateAxisAccel(0, 0, offsetX, scaleX);
  waitForUser("Теперь переверните датчик так, чтобы ось X была направлена ВНИЗ. Нажмите Enter...");
  calibrateAxisAccel(1, 0, offsetX, scaleX); // Второй проход для оси X, результат перезапишется
  printCalibrationResultAccel("X", offsetX, scaleX);

  // 4. Калибровка оси Y
  Serial.println(F("\n--- КАЛИБРОВКА ОСИ Y ---"));
  waitForUser("Положите датчик так, чтобы ось Y была направлена ВВЕРХ. Нажмите Enter...");
  calibrateAxisAccel(0, 1, offsetY, scaleY);
  waitForUser("Теперь переверните датчик так, чтобы ось Y была направлена ВНИЗ. Нажмите Enter...");
  calibrateAxisAccel(1, 1, offsetY, scaleY);
  printCalibrationResultAccel("Y", offsetY, scaleY);

  // 5. Калибровка оси Z
  Serial.println(F("\n--- КАЛИБРОВКА ОСИ Z ---"));
  waitForUser("Положите датчик так, чтобы ось Z была направлена ВВЕРХ. Нажмите Enter...");
  calibrateAxisAccel(0, 2, offsetZ, scaleZ);
  waitForUser("Теперь переверните датчик так, чтобы ось Z была направлена ВНИЗ. Нажмите Enter...");
  calibrateAxisAccel(1, 2, offsetZ, scaleZ);
  printCalibrationResultAccel("Z", offsetZ, scaleZ);

  // 6. Вывод итоговых коэффициентов в формате для копирования
  Serial.println(F("\n=== ИТОГОВЫЕ КОЭФФИЦИЕНТЫ ==="));
  Serial.println(F("Скопируйте эти значения и вставьте в ваш класс:"));
  Serial.print(F("offsets[0] = ")); Serial.print(offsetX, 6); Serial.println(F("; // X offset"));
  Serial.print(F("offsets[1] = ")); Serial.print(offsetY, 6); Serial.println(F("; // Y offset"));
  Serial.print(F("offsets[2] = ")); Serial.print(offsetZ, 6); Serial.println(F("; // Z offset"));
  Serial.print(F("scales[0]  = ")); Serial.print(scaleX, 6); Serial.println(F(";  // X scale"));
  Serial.print(F("scales[1]  = ")); Serial.print(scaleY, 6); Serial.println(F(";  // Y scale"));
  Serial.print(F("scales[2]  = ")); Serial.print(scaleZ, 6); Serial.println(F(";  // Z scale"));
}

void calibrateGyroscope() {
  float offset_gyro[3];
  Serial.println(F("\n--- КАЛИБРОВКА ГИРОСКОПА ---"));
  waitForUser("Положите датчик НЕПОДВИЖНО. Нажмите Enter...");
  readAverageGyro(offset_gyro, numReadings);

  Serial.println(F("\n=== ИТОГОВЫЕ КОЭФФИЦИЕНТЫ ==="));
  Serial.println(F("Скопируйте эти значения и вставьте в ваш класс:"));
  Serial.print(F("offsets[0] = ")); Serial.print(offset_gyro[0], 6); Serial.println(F("; // X offset"));
  Serial.print(F("offsets[1] = ")); Serial.print(offset_gyro[1], 6); Serial.println(F("; // Y offset"));
  Serial.print(F("offsets[2] = ")); Serial.print(offset_gyro[2], 6); Serial.println(F("; // Z offset"));
}


// ============================================================
void loop() {
  // Ничего не делаем, калибровка выполняется один раз
}

// ============================================================
// Инициализация BMI160 и проверка связи
void initSensor() {
  Serial.print(F("Инициализация датчика BMI160..."));
  // Попытка инициализации по I2C. Адрес по умолчанию: 0x68 или 0x69
  int err = IMU.init(calibration, 0x69);
  if (err != 0) {
    Serial.println(F(" [ОШИБКА]"));
    Serial.println(F("Проверьте подключение датчика и адрес I2C."));
    while (true) {delay(10);} // Останавливаем выполнение
  }
  Serial.println(F(" [ОК]"));
  
  delay(100);
}

// ============================================================
// Ожидание нажатия Enter от пользователя
void waitForUser(const char* message) {
  Serial.println(message);
  while (Serial.available() == 0) {
    delay(10);
  }
  while (Serial.available() > 0) {
    Serial.read(); // Очищаем буфер
  }
  Serial.println(F("Начинаю измерения..."));
  delay(500); // Небольшая задержка для стабилизации датчика
}

// ============================================================
// Чтение и усреднение данных с датчика
void readAverageAccel(float* avg, int samples) {
  float sum[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < samples; i++) {
    IMU.updateGyroAccel();
    IMU.getAccel(&accel);
    
    sum[0] += accel.accelX;
    sum[1] += accel.accelY;
    sum[2] += accel.accelZ;
    delay(5); // Небольшая пауза между измерениями
  }
  avg[0] = sum[0] / samples;
  avg[1] = sum[1] / samples;
  avg[2] = sum[2] / samples;
}

void readAverageGyro(float* avg, int samples) {
  float sum[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < samples; i++) {
    IMU.updateGyroAccel();
    IMU.getGyro(&gyro);
    
    sum[0] += gyro.gyroX;
    sum[1] += gyro.gyroY;
    sum[2] += gyro.gyroZ;
    delay(5); // Небольшая пауза между измерениями
  }
  avg[0] = sum[0] / samples;
  avg[1] = sum[1] / samples;
  avg[2] = sum[2] / samples;
}

// ============================================================
// Калибровка одной оси: вычисление смещения и масштаба
// Параметры:
//   orientation: 0 - вверх, 1 - вниз
//   axis: 0 - X, 1 - Y, 2 - Z
//   offset: ссылка на переменную для сохранения смещения
//   scale: ссылка на переменную для сохранения масштаба
void calibrateAxisAccel(int orientation, int axis, float& offset, float& scale) {
  float avg[3];
  readAverageAccel(avg, numReadings);

  
  // Сохраняем текущие значения в локальные переменные
  // Если это первый вызов (для верха), то offset и scale ещё не определены.
  // В этом случае мы просто сохраним измеренное значение.
  static float upValue[3] = {0.0f, 0.0f, 0.0f};
  
  if (orientation == 0) { // Положение "вверх"
    upValue[axis] = avg[axis];
    Serial.print(F("Измерено (вверх): ")); Serial.println(upValue[axis], 6);
  } else { // Положение "вниз"
    float downValue = avg[axis];
    Serial.print(F("Измерено (вниз): ")); Serial.println(downValue, 6);
    
    // Расчёт смещения и масштаба
    offset = (upValue[axis] + downValue) / 2.0f;
    scale = 2.0f / (upValue[axis] - downValue);
    // Если хотите получить масштаб для преобразования в g, используйте:
    // scale = 2.0f / (upValue[axis] - downValue);
  }
}

// ============================================================
// Вывод результатов для одной оси
void printCalibrationResultAccel(const char* axisName, float offset, float scale) {
  Serial.print(F("Результаты для оси ")); Serial.print(axisName); Serial.println(F(":"));
  Serial.print(F("  Offset: ")); Serial.println(offset, 6);
  Serial.print(F("  Scale:  ")); Serial.println(scale, 6);
  Serial.println();
}