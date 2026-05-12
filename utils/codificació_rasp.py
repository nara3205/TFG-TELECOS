import os
import time
import smbus
import RPi.GPIO as GPIO
import threading

# ---------------- GPIO ----------------
GPIO.setmode(GPIO.BCM)
TX_PIN = 27
GPIO.setup(TX_PIN, GPIO.OUT)
BUTTON_PIN = 17
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

BIT_TIME = 0.02   # més lent (~50 bps) per visualitzar bé

# ---------------- I2C ----------------
bus = smbus.SMBus(1)
address = 0x57

def write(reg, value):
    bus.write_byte_data(address, reg, value)
    
def shutdown_system(channel):
    time.sleep(2)
    if GPIO.input(BUTTON_PIN) == GPIO.LOW:
        print("Apagant sistema...")
        os.system("sudo shutdown -h now")

def read_fifo():
    data = bus.read_i2c_block_data(address, 0x07, 6)
    red = (data[0]<<16) | (data[1]<<8) | data[2]
    ir  = (data[3]<<16) | (data[4]<<8) | data[5]
    return red, ir

# ---------------- BPM ----------------
def calculate_bpm(ir_values, sample_rate=20):
    if len(ir_values) < 3:
        return 0

    peaks = []
    min_distance = int(0.5 * sample_rate)
    last_peak = -min_distance

    for i in range(1, len(ir_values)-1):
        if ir_values[i] > ir_values[i-1] and ir_values[i] > ir_values[i+1] and ir_values[i] > 50000:
            if i - last_peak >= min_distance:
                peaks.append(i)
                last_peak = i

    if len(peaks) < 2:
        return 0

    intervals = [(peaks[i+1]-peaks[i]) for i in range(len(peaks)-1)]
    avg_interval = sum(intervals) / float(len(intervals))
    bpm = 60 * sample_rate / avg_interval

    return int(bpm)

# ---------------- SENSOR SETUP ----------------
def setup_sensor():
    write(0x09, 0x40)
    time.sleep(0.1)
    write(0x02,0xC0)
    write(0x04,0x00)
    write(0x05,0x00)
    write(0x06,0x00)
    write(0x0A,0x27)
    write(0x0C,0x24)
    write(0x0D,0x24)
    write(0x09,0x03)

# ---------------- PROTOCOLO ----------------
def send_bit(bit):
    if bit == "1":
        GPIO.output(TX_PIN, 1)
        time.sleep(0.01)
    else:
        GPIO.output(TX_PIN, 0)
        time.sleep(0.01)

def send_frame(bpm):
	data = format(bpm, '08b')
	start = "101010"
	preamble = format(len(data), '08b')
	frame = start + preamble + data
	print("FRAME:", frame)
	for bit in frame:
		send_bit(bit)	
		time.sleep(0.01)
	GPIO.output(TX_PIN, 0)

# ---------------- VARIABLES COMPARTIDES ----------------
ir_buffer = []
buffer_size = 300
sample_rate = 20
bpm_to_send = 0
lock = threading.Lock()  # protegeix la variable bpm_to_send

# ---------------- THREADS ----------------
def sensor_thread():
    global bpm_to_send
    while True:
        try:
            red, ir = read_fifo()
        except Exception as e:
            print("Error I2C:", e)
            continue

        ir_buffer.append(ir)
        if len(ir_buffer) > buffer_size:
            ir_buffer.pop(0)

        bpm = calculate_bpm(ir_buffer, sample_rate)

        with lock:
            bpm_to_send = bpm

        print("Red:", red, "IR:", ir, "BPM:", bpm)
        time.sleep(1.0 / sample_rate)

def sender_thread():
    global bpm_to_send
    while True:
        with lock:
            bpm = bpm_to_send
        if bpm > 0:
            send_frame(bpm)
        time.sleep(5)  # temps entre trames

# ---------------- MAIN ----------------
setup_sensor()
     
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=shutdown_system, bouncetime=300)

try:
	t1 = threading.Thread(target=sensor_thread)
	t1.setDaemon(True)
	t2 = threading.Thread(target=sender_thread)
	t2.setDaemon(True)
	t1.start()
	t2.start()
	
	# Manté el programa actiu
	while True:
        	time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
    print("Programa finalitzat")
