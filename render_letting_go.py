import subprocess, math, sys
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageOps

U = "/root/.claude/uploads/2e7a7f4b-7177-59f4-8794-1c1edd170938/"
SONG = U + "e5015e9b-ScreenRecording_10-02-2026_12-21-10_1.mp4"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/user/Edit-de-videos/edit_letting_go.mp4"
PREVIEW = len(sys.argv) > 2

W, H, FPS = 1080, 1920, 30
A0 = 7.56                      # clean continuous part of the song starts here
DUR = 26.77 - A0
BEAT0, BP = 0.14, 0.5576       # first beat (relative) and beat period (~107.7 BPM)
beat = lambda k: BEAT0 + BP * k
CUTS = [0.0] + [beat(k) for k in (4, 9, 14, 19, 24, 29)] + [DUR]
XF = 0.22                      # half-length of crossfade

# ---------- grading ----------
def grade(img, gamma=1.0, gain=1.0, sat=1.0):
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    a = np.clip(a * gain, 0, 1) ** gamma
    lum = (a @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    a = lum + (a - lum) * sat
    # soft S-curve
    a = np.clip(a, 0, 1)
    a = a + 0.25 * (a - a * a) * (2 * a - 1)
    # split tone: teal shadows, warm highlights
    sh = (1 - lum) ** 2
    hi = lum ** 2
    a = a + sh * np.array([-0.015, 0.012, 0.03], np.float32) + hi * np.array([0.04, 0.015, -0.03], np.float32)
    # faded film blacks
    a = a * 0.93 + 0.035
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

def load(name, crop=None, **g):
    im = ImageOps.exif_transpose(Image.open(U + name)).convert("RGB")
    if crop:
        im = im.crop(crop)
    return grade(im, **g)

# order tells a little story: the city at night -> dinner -> home -> a blurry memory
SEGS = [
    # image, zoom start/end, center start/end (fractions of image)
    dict(im=load("e95bdc86-image.jpg", gamma=0.85, gain=1.1, sat=1.1), z=(1.12, 1.0), c=((0.55, 0.50), (0.50, 0.48))),
    dict(im=load("e1c0c900-image.jpg", gamma=0.9, sat=1.05), z=(1.45, 1.6), c=((0.40, 0.44), (0.58, 0.43))),
    dict(im=load("91dfd016-image.jpg", gamma=0.95, sat=1.0), z=(1.0, 1.04), c=((0.20, 0.5), (0.80, 0.5))),
    dict(im=load("b32b38fe-image.jpg", gamma=0.5, gain=1.9, sat=1.2), z=(1.75, 1.55), c=((0.70, 0.42), (0.52, 0.45))),
    dict(im=load("b9fc4959-image.jpg", gamma=1.0, sat=0.95), z=(1.04, 1.0), c=((0.22, 0.5), (0.62, 0.5))),
    dict(im=load("17a99166-image.jpg", gamma=1.0, sat=1.0), z=(1.10, 1.0), c=((0.5, 0.42), (0.5, 0.52))),
    dict(im=load("b57e730c-image.png", crop=(0, 150, 1179, 1860), gamma=1.0, sat=1.1), z=(1.0, 1.12), c=((0.5, 0.55), (0.5, 0.5))),
]

def ease(p):
    return p * p * (3 - 2 * p)

def render_seg(i, t):
    s = SEGS[i]
    im = s["im"]
    t0, t1 = CUTS[i] - XF, CUTS[i + 1] + XF
    p = ease(min(max((t - t0) / (t1 - t0), 0), 1)) * 0.6 + 0.4 * min(max((t - t0) / (t1 - t0), 0), 1)
    z = s["z"][0] + (s["z"][1] - s["z"][0]) * p
    # punch-in right after the cut, decaying
    dt = t - CUTS[i]
    if i > 0 and dt > -XF:
        z *= 1 + 0.045 * math.exp(-max(dt + XF, 0) / 0.35)
    iw, ih = im.size
    sc = max(W / iw, H / ih) * z
    cx = (s["c"][0][0] + (s["c"][1][0] - s["c"][0][0]) * p) * iw
    cy = (s["c"][0][1] + (s["c"][1][1] - s["c"][0][1]) * p) * ih
    hw, hh = W / 2 / sc, H / 2 / sc
    cx = min(max(cx, hw), iw - hw)
    cy = min(max(cy, hh), ih - hh)
    k = 1 / sc
    fr = im.transform((W, H), Image.AFFINE, (k, 0, cx - hw, 0, k, cy - hh), resample=Image.BICUBIC)
    return np.asarray(fr).astype(np.float32) / 255.0

# ---------- overlays ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.70)) ** 2)
VIG = (1 - 0.55 * np.clip(r - 0.35, 0, 1) ** 1.6)[..., None]

rng = np.random.default_rng(7)
GRAIN = []
for _ in range(10):
    n = rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32)
    n = np.asarray(Image.fromarray(((n * 40) + 128).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32)
    GRAIN.append(((n - 128) / 40.0 * 0.028)[..., None])

def leak(t, tc, warm):
    # soft light leak sweeping across during transitions
    q = (t - (tc - 0.45)) / 1.1
    if q < 0 or q > 1:
        return None
    amp = math.sin(math.pi * q) ** 2 * 0.24
    lx = -0.3 + 1.6 * q
    sw, shh = 135, 240
    gy, gx = np.mgrid[0:shh, 0:sw].astype(np.float32)
    d = ((gx / sw - lx) / 0.45) ** 2 + ((gy / shh - (0.3 + 0.3 * q)) / 0.6) ** 2
    m = np.exp(-d)
    col = np.array(warm, np.float32)
    small = (m[..., None] * col * amp)
    big = np.asarray(Image.fromarray((small.clip(0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    return big

LEAKS = [(1.0, 0.55, 0.25), (1.0, 0.35, 0.45), (1.0, 0.7, 0.4)]

def text_layer(lines):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for txt, font, size, y, spacing, alpha in lines:
        f = ImageFont.truetype(font, size)
        widths = [d.textlength(ch, font=f) for ch in txt]
        total = sum(widths) + spacing * (len(txt) - 1)
        x = (W - total) / 2
        for ch, w in zip(txt, widths):
            d.text((x, y), ch, font=f, fill=(255, 250, 244, alpha))
            x += w + spacing
    glow = layer.filter(ImageFilter.GaussianBlur(14))
    a = np.asarray(layer).astype(np.float32) / 255
    g = np.asarray(glow).astype(np.float32) / 255
    return a, g

SERIF_I = "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
TITLE = text_layer([
    ("ALMOST BARELY", SERIF, 30, 300, 14, 210),
    ("letting go", SERIF_I, 118, 345, 6, 255),
])
OUTRO = text_layer([
    ("letting go", SERIF_I, 64, 1590, 4, 255),
    ("ALMOST BARELY", SERIF, 22, 1675, 10, 190),
])

def over(fr, layer, op):
    if op <= 0:
        return fr
    a, g = layer
    fr = 1 - (1 - fr) * (1 - g[..., :3] * g[..., 3:] * 0.45 * op)   # soft glow (screen)
    al = a[..., 3:] * op
    return fr * (1 - al) + a[..., :3] * al

def window(t, a, b, fi, fo):
    if t < a or t > b:
        return 0.0
    return ease(min(1, (t - a) / fi, (b - t) / fo))

# ---------- render ----------
nframes = int(DUR * FPS)
if PREVIEW:
    frames_idx = range(0, nframes, 6)
else:
    frames_idx = range(nframes)

cmd = ["ffmpeg", "-v", "error", "-y",
       "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS / (6 if PREVIEW else 1)), "-i", "-",
       "-ss", str(A0), "-t", str(DUR), "-i", SONG,
       "-map", "0:v", "-map", "1:a",
       "-af", f"afade=t=in:d=0.25,afade=t=out:st={DUR - 2.6:.2f}:d=2.6",
       "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT]
ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

for n in frames_idx:
    t = n / FPS
    # which segment(s) are visible
    fr = None
    for i in range(len(SEGS)):
        a, b = CUTS[i] - XF, CUTS[i + 1] + XF
        if a <= t <= b:
            layer = render_seg(i, t)
            if fr is None:
                fr = layer
            else:
                w = ease((t - (CUTS[i] - XF)) / (2 * XF))
                fr = fr * (1 - w) + layer * w
    # bloom flash on cuts
    for k, tc in enumerate(CUTS[1:-1]):
        fl = math.exp(-((t - tc) / 0.12) ** 2) * 0.22
        if fl > 0.005:
            fr = fr + (1 - fr) * fl
        lk = leak(t, tc, LEAKS[k % 3])
        if lk is not None:
            fr = 1 - (1 - fr) * (1 - lk)
    # dreamy blur + fade on the ending
    if t > DUR - 2.4:
        q = (t - (DUR - 2.4)) / 2.4
        img = Image.fromarray((fr.clip(0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1 + 14 * q * q))
        fr = np.asarray(img).astype(np.float32) / 255
    fr = fr * VIG + GRAIN[n % len(GRAIN)]
    fr = over(fr, TITLE, window(t, 0.35, 2.25, 0.9, 0.5))
    fr = over(fr, OUTRO, window(t, DUR - 3.3, DUR - 0.5, 1.0, 0.8))
    # fade in from black / out to black
    fr = fr * min(1, t / 0.5, max(0, (DUR - t) / 1.0))
    ff.stdin.write((fr.clip(0, 1) * 255).astype(np.uint8).tobytes())
    if n % 60 == 0:
        print(f"{t:.1f}s", flush=True)

ff.stdin.close()
ff.wait()
print("done", OUT)
