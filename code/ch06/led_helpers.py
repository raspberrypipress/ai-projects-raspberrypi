"""
led_helpers.py

Small MicroPython helpers for NeoPixels.

Exports:
    hsv_to_rgb(h, s=1.0, v=1.0)
    sparkle(np, led_count, total_time, max_sparkle_duration, min_sparkle_duration, fade)

Behavior of sparkle():
- Spawns sparkles for `total_time` seconds
- After `total_time`, no new sparkles are created
- Existing sparkles continue fading out naturally
- The function returns only after all sparkles have fully faded out
"""

try:
    import urandom as random
except ImportError:
    import random

from time import sleep, ticks_ms


def _clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _randbelow(n):
    """Return an integer in [0, n). Works on MicroPython and CPython."""
    if n <= 0:
        return 0

    if hasattr(random, "getrandbits"):
        bits = 1
        while (1 << bits) < n:
            bits += 1
        while True:
            r = random.getrandbits(bits)
            if r < n:
                return r

    if hasattr(random, "randrange"):
        return random.randrange(n)

    return int(random.random() * n)


def _normalize_hue(h):
    """
    Convert hue input to 0.0..1.0.

    Accepted forms:
    - 0.0..1.0   -> unchanged
    - 0..255     -> treated as 8-bit color wheel
    - 0..360     -> treated as degrees
    """
    if h is None:
        return 0.0

    if isinstance(h, float) and 0.0 <= h <= 1.0:
        return h

    if 0 <= h <= 255:
        return h / 255.0

    h = h % 360
    return h / 360.0


def hsv_to_rgb(h, s=1.0, v=1.0):
    """
    Convert HSV to an (r, g, b) tuple of 0..255 ints.

    Parameters:
        h: hue as 0.0..1.0, 0..255, or degrees
        s: saturation, 0.0..1.0
        v: brightness/value, 0.0..1.0
    """
    h = _normalize_hue(h)
    s = _clamp(s, 0.0, 1.0)
    v = _clamp(v, 0.0, 1.0)

    if s == 0.0:
        x = int(v * 255)
        return (x, x, x)

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return (int(r * 255), int(g * 255), int(b * 255))


class _SparkleEvent:
    def __init__(self, idx, color, duration_ms):
        self.idx = idx
        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.duration_ms = duration_ms
        self.start_ms = ticks_ms()

    def brightness(self, now_ms):
        age = now_ms - self.start_ms
        if age < 0:
            age = 0
        if age >= self.duration_ms:
            return 0.0
        return 1.0 - (age / self.duration_ms)

    def alive(self, now_ms):
        return (now_ms - self.start_ms) < self.duration_ms


def sparkle(np, led_count, total_time, max_sparkle_duration, min_sparkle_duration, fade):
    """
    Run a sparkle animation on a NeoPixel strip.

    Parameters:
        np: NeoPixel object
        led_count: number of LEDs to use from the start of the strip
        total_time: total time in seconds during which new sparkles are spawned
        max_sparkle_duration: maximum lifetime of a sparkle in seconds
        min_sparkle_duration: minimum lifetime of a sparkle in seconds
        fade: overall brightness multiplier, usually 0.0..1.0

    Behavior:
        - New sparkles are created only during total_time
        - Existing sparkles continue to fade after total_time
        - Function returns only when all sparkles have faded out
    """
    try:
        strip_len = len(np)
    except TypeError:
        strip_len = led_count

    n = led_count if led_count < strip_len else strip_len
    if n <= 0:
        return

    total_ms = int(total_time * 1000)
    min_ms = int(min_sparkle_duration * 1000)
    max_ms = int(max_sparkle_duration * 1000)
    fade = _clamp(fade, 0.0, 1.0)

    if total_ms <= 0:
        for i in range(n):
            np[i] = (0, 0, 0)
        np.write()
        return

    if min_ms < 1:
        min_ms = 1
    if max_ms < min_ms:
        max_ms = min_ms

    frame_ms = 40
    sleep_s = frame_ms / 1000.0

    for i in range(n):
        np[i] = (0, 0, 0)
    np.write()

    active = []
    start_ms = ticks_ms()
    last_spawn_ms = start_ms
    spawning = True

    avg_ms = (min_ms + max_ms) // 2
    divisor = n // 3
    if divisor < 2:
        divisor = 2
    spawn_interval_ms = avg_ms // divisor
    if spawn_interval_ms < 30:
        spawn_interval_ms = 30
    if spawn_interval_ms > 180:
        spawn_interval_ms = 180

    while True:
        now = ticks_ms()

        if spawning and (now - start_ms) >= total_ms:
            spawning = False

        if spawning:
            while (now - last_spawn_ms) >= spawn_interval_ms:
                idx = _randbelow(n)

                duration_ms = min_ms
                if max_ms > min_ms:
                    duration_ms += _randbelow(max_ms - min_ms + 1)

                hue = _randbelow(256)
                value = 0.6 + (_randbelow(103) / 255.0)
                color = hsv_to_rgb(hue, 1.0, value)

                active.append(_SparkleEvent(idx, color, duration_ms))
                last_spawn_ms += spawn_interval_ms

        kept = []
        for s in active:
            if s.alive(now):
                kept.append(s)
        active = kept

        if (not spawning) and (not active):
            break

        for i in range(n):
            np[i] = (0, 0, 0)

        for s in active:
            b = s.brightness(now) * fade
            if b <= 0.0:
                continue

            r = int(s.r * b)
            g = int(s.g * b)
            bl = int(s.b * b)

            current = np[s.idx]
            nr = current[0] + r
            ng = current[1] + g
            nb = current[2] + bl

            if nr > 255:
                nr = 255
            if ng > 255:
                ng = 255
            if nb > 255:
                nb = 255

            np[s.idx] = (nr, ng, nb)

        np.write()
        sleep(sleep_s)

    for i in range(n):
        np[i] = (0, 0, 0)
    np.write()


