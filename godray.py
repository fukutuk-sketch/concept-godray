"""godray.py - ゴッドレイ (crepuscular rays) エフェクト

画像の明るい部分を抽出し、光源位置へ向かう放射状ブラーで光条を作って合成する。
(GPU Gems 3 "Volumetric Light Scattering as a Post-Process" 方式のCPU/numpy実装)

使い方:
    python godray.py input.png output.png --light 0.7,0.15
    python godray.py --demo demo.png          # 合成シーンでデモ生成
"""

import argparse

import numpy as np
from PIL import Image


def luminance(img):
    return img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bright_pass(img, threshold=0.7, softness=0.15):
    """しきい値以上の明るい部分だけを残した画像 (光線の元になる)。"""
    lum = luminance(img)
    mask = smoothstep(threshold - softness, threshold + softness, lum)
    return img * mask[..., None]


def sample_bilinear(img, xs, ys):
    """画像 (H,W,C) を座標 (xs, ys) でバイリニアサンプリング。範囲外は端に張り付く。"""
    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def radial_blur(src, light, samples=64, density=0.9, decay=0.96, weight=1.0):
    """各ピクセルから光源へ向かって samples 回サンプリングし、減衰させながら加算する。

    light: (x, y) をピクセル座標で。
    density: 1.0 で光源まで全距離をたどる。小さいほど光条が短い。
    decay: 1サンプルごとの減衰率。
    """
    h, w = src.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (light[0] - gx) * density / samples
    dy = (light[1] - gy) * density / samples

    out = np.zeros_like(src)
    illum = 1.0
    cx, cy = gx.copy(), gy.copy()
    for _ in range(samples):
        cx += dx
        cy += dy
        out += sample_bilinear(src, cx, cy) * illum * weight
        illum *= decay
    return out / samples


def godray(
    img,
    light=(0.5, 0.1),
    threshold=0.7,
    softness=0.15,
    samples=64,
    density=0.9,
    decay=0.96,
    weight=1.0,
    exposure=6.0,
    intensity=1.0,
    tint=(1.0, 0.95, 0.8),
    downscale=2,
):
    """img: float32 (H,W,3) 0..1。光源位置 light は画像サイズに対する相対座標 (0..1)。"""
    h, w = img.shape[:2]

    # 重い放射ブラーは縮小して計算し、最後に元サイズへ戻す
    sw, sh = max(w // downscale, 1), max(h // downscale, 1)
    small = np.asarray(
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
            (sw, sh), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0

    src = bright_pass(small, threshold, softness)
    light_px = (light[0] * (sw - 1), light[1] * (sh - 1))
    rays = radial_blur(src, light_px, samples, density, decay, weight) * exposure

    rays_img = Image.fromarray((np.clip(rays, 0, 1) * 255).astype(np.uint8)).resize(
        (w, h), Image.BICUBIC
    )
    rays = np.asarray(rays_img, dtype=np.float32) / 255.0

    # 色味を付けて加算合成 (screen)
    rays = rays * np.array(tint, dtype=np.float32) * intensity
    return 1.0 - (1.0 - np.clip(img, 0, 1)) * (1.0 - np.clip(rays, 0, 1))


def make_demo_scene(w=960, h=540, light=(0.7, 0.12)):
    """太陽と、光を遮る木々のシルエットがある合成シーン。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = yy / h
    sky = np.stack([0.25 + 0.35 * t, 0.35 + 0.30 * t, 0.65 - 0.10 * t], axis=-1)

    lx, ly = light[0] * w, light[1] * h
    d = np.hypot(xx - lx, yy - ly)
    sun = np.exp(-((d / 28.0) ** 4))[..., None]
    glow = np.exp(-d / 160.0)[..., None] * 0.35
    img = sky + glow * np.array([1.0, 0.85, 0.6]) + sun * np.array([1.0, 0.98, 0.9])

    # 木の幹と地面のシルエット
    rng = np.random.default_rng(7)
    occ = np.zeros((h, w), dtype=np.float32)
    for x in rng.integers(0, w, 14):
        width = rng.integers(14, 36)
        top = rng.integers(0, int(h * 0.45))
        occ[top:, max(x - width // 2, 0) : x + width // 2] = 1.0
    for x in rng.integers(0, w, 10):  # 枝葉の塊
        r = rng.integers(40, 90)
        y = rng.integers(0, int(h * 0.4))
        occ = np.maximum(occ, ((xx - x) ** 2 + (yy - y) ** 2 < r * r).astype(np.float32))
    occ[int(h * 0.85) :, :] = 1.0
    img = img * (1 - occ[..., None]) + np.array([0.03, 0.05, 0.04]) * occ[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def parse_pair(text):
    x, y = (float(v) for v in text.split(","))
    return x, y


def parse_color(text):
    r, g, b = (float(v) for v in text.split(","))
    return r, g, b


def main():
    p = argparse.ArgumentParser(description="ゴッドレイ (光条) エフェクト")
    p.add_argument("input", nargs="?", help="入力画像")
    p.add_argument("output", nargs="?", help="出力画像")
    p.add_argument("--demo", metavar="OUT", help="合成シーンでデモ画像を出力")
    p.add_argument("--light", type=parse_pair, default=(0.7, 0.12), help="光源位置 x,y (0..1)")
    p.add_argument("--threshold", type=float, default=0.7, help="明部抽出しきい値")
    p.add_argument("--samples", type=int, default=64, help="放射ブラーのサンプル数")
    p.add_argument("--density", type=float, default=0.9, help="光条の長さ (0..1)")
    p.add_argument("--decay", type=float, default=0.96, help="サンプルごとの減衰率")
    p.add_argument("--weight", type=float, default=1.0, help="サンプル重み")
    p.add_argument("--exposure", type=float, default=6.0, help="光条の明るさ補正")
    p.add_argument("--intensity", type=float, default=1.0, help="合成の強さ")
    p.add_argument("--tint", type=parse_color, default=(1.0, 0.95, 0.8), help="光の色 r,g,b (0..1)")
    p.add_argument("--downscale", type=int, default=2, help="計算用の縮小率 (大きいほど高速)")
    args = p.parse_args()

    if args.demo:
        img = make_demo_scene(light=args.light)
        out_path = args.demo
    elif args.input and args.output:
        img = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.float32) / 255.0
        out_path = args.output
    else:
        p.error("input と output、または --demo OUT を指定してください")

    result = godray(
        img,
        light=args.light,
        threshold=args.threshold,
        samples=args.samples,
        density=args.density,
        decay=args.decay,
        weight=args.weight,
        exposure=args.exposure,
        intensity=args.intensity,
        tint=args.tint,
        downscale=max(args.downscale, 1),
    )
    Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
