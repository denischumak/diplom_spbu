const unsigned int MAX_COUNT = 10000;

const int pinRead = A1;

void setup() {
  Serial.begin(115200);
  pinMode(pinRead, INPUT);
}

unsigned int ttime = 0;
unsigned int count = 0;


// 60HZ
void loop() {
  int raw = analogRead(pinRead);
  Serial.println(raw);
  if (++count >= MAX_COUNT) {
    while(1);
  }
  delayMicroseconds(16600);
}