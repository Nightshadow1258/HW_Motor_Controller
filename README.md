# Motor Controller Board - Hardware
Hardware design files and documentation for the 3-phase custom Motor Controller Board (Hardware Revision v1.0). This board is designed for precise BLDC motor control, featuring integrated current sensing, reverse polarity protection, dynamic gate driving, and onboard networking interfaces.
## Board Status
- Current Hardware Version: v1.0
- Status: Tested & Verified (Timer PWM generation, ADC sampling, and hardware bringup verified)
## Key Hardware Components
- MCU: STM32H750VB — High-performance ARM Cortex-M7 Microcontroller (16-bit ADCs, TIM1 Advanced Timer)
- Gate Driver Unit (GDU): DRV8334 — 3-Phase Smart Gate Driver with integrated Current Sense Amplifiers (CSA)
- Ethernet PHY: LAN8742A — 10/100 Ethernet Transceiver for high-speed network communication
- CAN Transceiver: TJA1043T — High-speed CAN Transceiver with sleep mode and wake-up capabilities
- Reverse Polarity Protection (RPP): LM74700 — Ideal Diode Controller for low-loss battery/power reverse polarity protection
- Thermal Sensor: NTCS0603E3103FLT — 10kΩ NTC Thermistor for board temperature monitoring
## Repository Contents
- `/KiCAD` - source schematics files
- `/Documentation/Schematics` — schematic PDF export

# Related Resources
- Firmware Repository: [Motor Controller Board Firmware](https://github.com/Nightshadow1258/FW_STM32H750VB_DRV8334)
- Blog Posts:
	- [Part I - Hardware Design & Bringup](https://blog.preisch.cloud/diy-motor-controller-part-i-hardware-design-bringup/)

# License
This hardware design is licensed under the CERN-OHL-P v2
