/* * -------------------------------------------------------------------
 * MODIFIED IMPLEMENTATION NOTICE:
 * This file is a modified version of the original source code from 
 * the FastIMU library (https://github.com/LiquidCGS/FastIMU).
 * -------------------------------------------------------------------
 */

#pragma once

#ifndef _F_IMUUtils_H_
#define _F_IMUUtils_H_

#include <Wire.h>
#include <Arduino.h>

// Declare the functions
void writeByteI2C(TwoWire &wire, uint8_t address, uint8_t subAddress, uint8_t data);
uint8_t readByteI2C(TwoWire &wire, uint8_t address, uint8_t subAddress);
void readBytesI2C(TwoWire &wire, uint8_t address, uint8_t subAddress, uint8_t count, uint8_t *dest);

#endif
