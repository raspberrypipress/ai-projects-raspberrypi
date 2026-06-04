import machine
i2c = machine.I2C(0, scl=machine.Pin(17), sda=machine.Pin(16))

print('Scanning i2c bus...')
devices = i2c.scan()

if len(devices) == 0:
    print("No i2c devices found!")
else:
    print(f"Found {len(devices)} i2c devices.")

    for device in devices:  
        print(f"Decimal addr: {device} | Hex addr: {hex(device)}")
