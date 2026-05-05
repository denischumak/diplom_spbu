#include "BMI160.hpp"

int BMI160::init(calData cal, uint8_t address)
{
    // initialize address variable and calibration data.
    IMUAddress = address;

    if (cal.valid == false)
    {
        calibration.valid = true;
        for (int i = 0; i < 3; ++i)
        {
            calibration.accelBias[i] = 0;
            calibration.gyroBias[i] = 0;
            calibration.accelScale[i] = 1;
        }
    }
    else
    {
        calibration = cal;
    }

    if (!(readByteI2C(wire, IMUAddress, BMI160_CHIP_ID) == BMI160_CHIP_ID_DEFAULT_VALUE))
    {
        return -1;
    }

    // reset device
    writeByteI2C(wire, IMUAddress, BMI160_CMD, 0xB6); // Toggle softreset
    delay(100);                                       // wait for reset

    writeByteI2C(wire, IMUAddress, BMI160_CMD, 0x11); // Start up accelerometer
    delay(100);                                       // wait until they're done starting up...
    writeByteI2C(wire, IMUAddress, BMI160_CMD, 0x15); // Start up gyroscope
    delay(100);                                       // wait until they're done starting up...

    writeByteI2C(wire, IMUAddress, BMI160_ACC_RANGE, 0x0C); // Set up full scale Accel range. +-16G
    writeByteI2C(wire, IMUAddress, BMI160_GYR_RANGE, 0x00); // Set up full scale Gyro range. +-2000dps

    writeByteI2C(wire, IMUAddress, BMI160_ACC_CONF, 0x09); // Set Accel ODR to 200hz, BWP mode to Oversample 4,
    writeByteI2C(wire, IMUAddress, BMI160_GYR_CONF, 0x09); // Set Gyro ODR to 200hz, BWP mode to Oversample 4,

    aRes = 16.f / 32768.f;   // ares value for full range (16g) readings
    gRes = 2000.f / 32768.f; // gres value for full range (2000dps) readings
    return 0;
}

void BMI160::updateGyroAccel()
{
    int16_t IMUCount[6]; // used to read all 16 bytes at once from the accel/gyro
    uint8_t rawData[12]; // x/y/z accel register data stored here

    readBytesI2C(wire, IMUAddress, BMI160_GYR_X_L, 12, &rawData[0]); // Read the 12 raw data registers into data array

    IMUCount[0] = ((int16_t)rawData[1] << 8) | rawData[0]; // Turn the MSB and LSB into a signed 16-bit value
    IMUCount[1] = ((int16_t)rawData[3] << 8) | rawData[2];
    IMUCount[2] = ((int16_t)rawData[5] << 8) | rawData[4];
    IMUCount[3] = ((int16_t)rawData[7] << 8) | rawData[6];
    IMUCount[4] = ((int16_t)rawData[9] << 8) | rawData[8];
    IMUCount[5] = ((int16_t)rawData[11] << 8) | rawData[10];

    float ax, ay, az, gx, gy, gz;

    // Calculate the accel value into actual g's per second
    accel.accelX = ((float)IMUCount[3] * aRes - calibration.accelBias[0]) * calibration.accelScale[0];
    accel.accelY = ((float)IMUCount[4] * aRes - calibration.accelBias[1]) * calibration.accelScale[1];
    accel.accelZ = ((float)IMUCount[5] * aRes - calibration.accelBias[2]) * calibration.accelScale[2];

    // Calculate the gyro value into actual degrees per second
    gyro.gyroX = (float)IMUCount[0] * gRes - calibration.gyroBias[0];
    gyro.gyroY = (float)IMUCount[1] * gRes - calibration.gyroBias[1];
    gyro.gyroZ = (float)IMUCount[2] * gRes - calibration.gyroBias[2];
}

void BMI160::getAccel(AccelData *out)
{
    memcpy(out, &accel, sizeof(accel));
}
void BMI160::getGyro(GyroData *out)
{
    memcpy(out, &gyro, sizeof(gyro));
}

float BMI160::getTemp()
{
    uint8_t buf[2];
    readBytesI2C(wire, IMUAddress, BMI160_TEMPERATURE_0, 2, &buf[0]);
    float temp = ((((int16_t)buf[1]) << 8) | buf[0]);
    return (temp / 512) + 23.f;
}

int BMI160::setAccelRange(int range)
{
    uint8_t c;
    if (range == 16)
    {
        aRes = 16.f / 32768.f; // ares value for full range (16g) readings
        c = 0x0C;
    }
    else if (range == 8)
    {
        aRes = 8.f / 32768.f; // ares value for range (8g) readings
        c = 0x08;
    }
    else if (range == 4)
    {
        aRes = 4.f / 32768.f; // ares value for range (4g) readings
        c = 0x05;
    }
    else if (range == 2)
    {
        aRes = 2.f / 32768.f; // ares value for range (2g) readings
        c = 0x03;
    }
    else
    {
        return -1;
    }
    writeByteI2C(wire, IMUAddress, BMI160_ACC_RANGE, c); // Write new ACCEL_CONFIG register value
    return 0;
}

int BMI160::setGyroRange(int range)
{
    uint8_t c;
    if (range == 2000)
    {
        gRes = 2000.f / 32768.f; // ares value for full range (2000dps) readings
        c = 0x00;
    }
    else if (range == 1000)
    {
        gRes = 1000.f / 32768.f; // ares value for range (1000dps) readings
        c = 0x01;
    }
    else if (range == 500)
    {
        gRes = 500.f / 32768.f; // ares value for range (500dps) readings
        c = 0x02;
    }
    else if (range == 250)
    {
        gRes = 250.f / 32768.f; // ares value for range (250dps) readings
        c = 0x03;
    }
    else if (range == 125)
    {
        gRes = 125.f / 32768.f; // ares value for range (250dps) readings
        c = 0x04;
    }
    else
    {
        return -1;
    }
    writeByteI2C(wire, IMUAddress, BMI160_GYR_RANGE, c); // Write new GYRO_CONFIG register value
    return 0;
}