//! JPEG превью с bbox-overlay и кроп снапшотов для тревог.

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};

use super::events::{PersonDet, TrackSnapshot};

/// Длинная сторона превью перед JPEG по умолчанию.
const DEFAULT_PREVIEW_ENCODE_MAX_EDGE: u32 = 960;
/// Качество JPEG по умолчанию. Держим умеренным: MJPEG кодируется постоянно.
const DEFAULT_PREVIEW_JPEG_QUALITY: u8 = 70;

/// JPEG вырезки по bbox события (с отступом), без overlay — для снапшота тревоги.
pub fn encode_alarm_crop_jpeg(
    bgr: &[u8],
    frame_w: u32,
    frame_h: u32,
    bbox: [f32; 4],
    pad_frac: f32,
    max_long_edge: u32,
    quality: u8,
) -> Option<Vec<u8>> {
    if frame_w == 0 || frame_h == 0 || bgr.len() < (frame_w as usize) * (frame_h as usize) * 3 {
        return None;
    }
    let mut x1 = bbox[0].min(bbox[2]);
    let mut y1 = bbox[1].min(bbox[3]);
    let mut x2 = bbox[0].max(bbox[2]);
    let mut y2 = bbox[1].max(bbox[3]);
    let bw = (x2 - x1).max(1.0);
    let bh = (y2 - y1).max(1.0);
    let pad_x = bw * pad_frac;
    let pad_y = bh * pad_frac;
    x1 = (x1 - pad_x).floor().max(0.0);
    y1 = (y1 - pad_y).floor().max(0.0);
    x2 = (x2 + pad_x).ceil().min(frame_w as f32);
    y2 = (y2 + pad_y).ceil().min(frame_h as f32);
    let ix1 = x1 as u32;
    let iy1 = y1 as u32;
    let ix2 = x2 as u32;
    let iy2 = y2 as u32;
    let cw = ix2.saturating_sub(ix1).max(1);
    let ch = iy2.saturating_sub(iy1).max(1);
    let mut rgb = vec![0u8; (cw * ch * 3) as usize];
    for row in 0..ch {
        let sy = iy1 + row;
        let src_row = (sy * frame_w + ix1) as usize * 3;
        let dst_row = row as usize * cw as usize * 3;
        for col in 0..cw {
            let si = src_row + col as usize * 3;
            let di = dst_row + col as usize * 3;
            if si + 2 < bgr.len() && di + 2 < rgb.len() {
                rgb[di] = bgr[si + 2];
                rgb[di + 1] = bgr[si + 1];
                rgb[di + 2] = bgr[si];
            }
        }
    }
    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(cw, ch, rgb)?;
    let img_scaled = if cw.max(ch) > max_long_edge {
        let scale = max_long_edge as f32 / cw.max(ch) as f32;
        let nw = (cw as f32 * scale).max(1.0) as u32;
        let nh = (ch as f32 * scale).max(1.0) as u32;
        image::imageops::resize(&img, nw, nh, FilterType::Triangle)
    } else {
        img
    };
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, quality);
    enc.encode_image(&img_scaled).ok()?;
    Some(jpeg)
}

/// BGR → RGB, даунскейл, bbox, JPEG.
///
/// **Треки объектов** на MJPEG: **оранжевый** (`unattended`) и **тревоги** (`alarm_*`).
/// `static` и `candidate` не рисуем — меньше шума на превью (см. `tracks_snapshot` в native).
///
/// **Люди** (`persons`): пурпурная рамка только при `draw_person_boxes == true` (`config.json` → `ui.show_persons`).
pub fn encode_preview_jpeg(
    bgr: &[u8],
    w: u32,
    h: u32,
    tracks: &[TrackSnapshot],
    persons: &[PersonDet],
    draw_person_boxes: bool,
) -> Option<Vec<u8>> {
    encode_preview_jpeg_with_options(
        bgr,
        w,
        h,
        tracks,
        persons,
        draw_person_boxes,
        DEFAULT_PREVIEW_ENCODE_MAX_EDGE,
        DEFAULT_PREVIEW_JPEG_QUALITY,
    )
}

/// BGR → RGB, даунскейл, bbox, JPEG с параметрами из runtime config.
pub fn encode_preview_jpeg_with_options(
    bgr: &[u8],
    w: u32,
    h: u32,
    tracks: &[TrackSnapshot],
    persons: &[PersonDet],
    draw_person_boxes: bool,
    max_long_edge: u32,
    jpeg_quality: u8,
) -> Option<Vec<u8>> {
    if w == 0 || h == 0 || bgr.len() < (w as usize) * (h as usize) * 3 {
        return None;
    }

    let max_long_edge = max_long_edge.clamp(320, 1920);
    let jpeg_quality = jpeg_quality.clamp(35, 95);

    let scale = if w.max(h) > max_long_edge {
        max_long_edge as f32 / w.max(h) as f32
    } else {
        1.0f32
    };

    // Сначала BGR→RGB на исходном разрешении (одно копирование), затем — качественный ресайз.
    let mut rgb_full = vec![0u8; (w * h * 3) as usize];
    let pixels = (w * h) as usize;
    for i in 0..pixels {
        let si = i * 3;
        let di = i * 3;
        rgb_full[di] = bgr[si + 2];
        rgb_full[di + 1] = bgr[si + 1];
        rgb_full[di + 2] = bgr[si];
    }

    let (ow, oh, mut rgb) = if scale < 1.0 {
        let nw = (w as f32 * scale).max(1.0).round() as u32;
        let nh = (h as f32 * scale).max(1.0).round() as u32;
        let img = ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, rgb_full)?;
        let resized = image::imageops::resize(&img, nw, nh, FilterType::Triangle);
        (nw, nh, resized.into_raw())
    } else {
        (w, h, rgb_full)
    };

    let iw = ow as i32;
    let ih = oh as i32;
    for t in tracks {
        if !track_visible_on_mjpeg_preview(t.state.as_str()) {
            continue;
        }
        let color = color_for_state(&t.state);
        let thickness = if t.alarm { 3 } else { 2 };
        let x1 = (t.bbox[0].min(t.bbox[2]) * scale) as i32;
        let y1 = (t.bbox[1].min(t.bbox[3]) * scale) as i32;
        let x2 = (t.bbox[0].max(t.bbox[2]) * scale) as i32;
        let y2 = (t.bbox[1].max(t.bbox[3]) * scale) as i32;
        draw_rect(&mut rgb, iw, ih, x1, y1, x2, y2, color, thickness);
    }
    // Люди поверх объектов: СИНИЙ бокс + подпись "PERSON #<id>".
    if draw_person_boxes {
        let person_color = [40, 110, 240];
        for p in persons {
            let x1 = (p.bbox[0].min(p.bbox[2]) * scale) as i32;
            let y1 = (p.bbox[1].min(p.bbox[3]) * scale) as i32;
            let x2 = (p.bbox[0].max(p.bbox[2]) * scale) as i32;
            let y2 = (p.bbox[1].max(p.bbox[3]) * scale) as i32;
            draw_rect(&mut rgb, iw, ih, x1, y1, x2, y2, person_color, 2);
            let label = format!("PERSON #{}", p.track_id);
            draw_label(&mut rgb, iw, ih, x1, y1, &label, person_color);
        }
    }

    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(ow, oh, rgb)?;
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, jpeg_quality);
    enc.encode_image(&img).ok()?;
    Some(jpeg)
}

/// На превью показываем объекты в двух состояниях: наблюдение (`static`/`unattended`)
/// и тревога (`alarm_*`). Сырые `candidate` (шумные motion-блобы) не рисуем.
fn track_visible_on_mjpeg_preview(state: &str) -> bool {
    matches!(
        state,
        "static" | "unattended" | "alarm_unattended" | "alarm_removed" | "alarm_missing"
    )
}

/// Два состояния объекта: ЖЁЛТЫЙ — система заметила и ждёт развития сценария;
/// КРАСНЫЙ — сценарий «оставлен/пропал/забрали» сработал (тревога).
fn color_for_state(state: &str) -> [u8; 3] {
    match state {
        "alarm_unattended" | "alarm_removed" | "alarm_missing" => [235, 30, 30], // красный
        _ => [255, 210, 0], // жёлтый (static / unattended / candidate)
    }
}

fn draw_rect(
    rgb: &mut [u8],
    w: i32,
    h: i32,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    color: [u8; 3],
    thickness: i32,
) {
    let x1 = x1.clamp(0, w - 1);
    let y1 = y1.clamp(0, h - 1);
    let x2 = x2.clamp(0, w - 1);
    let y2 = y2.clamp(0, h - 1);
    let put = |rgb: &mut [u8], px: i32, py: i32| {
        if px < 0 || py < 0 || px >= w || py >= h {
            return;
        }
        let idx = (py as usize * w as usize + px as usize) * 3;
        rgb[idx] = color[0];
        rgb[idx + 1] = color[1];
        rgb[idx + 2] = color[2];
    };
    for t in 0..thickness {
        for x in x1..=x2 {
            put(rgb, x, y1 + t);
            put(rgb, x, y2 - t);
        }
        for y in y1..=y2 {
            put(rgb, x1 + t, y);
            put(rgb, x2 - t, y);
        }
    }
}

/// Закрасить прямоугольник (подложка под подпись).
fn fill_rect(rgb: &mut [u8], w: i32, h: i32, x1: i32, y1: i32, x2: i32, y2: i32, color: [u8; 3]) {
    let xa = x1.clamp(0, w - 1);
    let xb = x2.clamp(0, w - 1);
    let ya = y1.clamp(0, h - 1);
    let yb = y2.clamp(0, h - 1);
    for y in ya..=yb {
        for x in xa..=xb {
            let idx = (y as usize * w as usize + x as usize) * 3;
            rgb[idx] = color[0];
            rgb[idx + 1] = color[1];
            rgb[idx + 2] = color[2];
        }
    }
}

/// Подпись над боксом: подложка цветом бокса + белый текст.
fn draw_label(rgb: &mut [u8], w: i32, h: i32, bx: i32, by: i32, text: &str, bg: [u8; 3]) {
    let s = 2i32;
    let cell = 6 * s;
    let tw = text.chars().count() as i32 * cell + 3;
    let th = 7 * s + 2;
    let mut ly = by - th;
    if ly < 0 {
        ly = by;
    }
    let lx = bx.max(0);
    fill_rect(rgb, w, h, lx, ly, lx + tw, ly + th, bg);
    draw_text(rgb, w, h, lx + 2, ly + 1, text, s, [255, 255, 255]);
}

/// Нарисовать строку встроенным 5x7 шрифтом.
fn draw_text(rgb: &mut [u8], w: i32, h: i32, x: i32, y: i32, text: &str, scale: i32, fg: [u8; 3]) {
    let mut cx = x;
    for ch in text.chars() {
        if let Some(g) = glyph(ch) {
            for (r, row) in g.iter().enumerate() {
                for (c, b) in row.bytes().enumerate() {
                    if b == b'#' {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let px = cx + c as i32 * scale + dx;
                                let py = y + r as i32 * scale + dy;
                                if px >= 0 && py >= 0 && px < w && py < h {
                                    let idx = (py as usize * w as usize + px as usize) * 3;
                                    rgb[idx] = fg[0];
                                    rgb[idx + 1] = fg[1];
                                    rgb[idx + 2] = fg[2];
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * scale;
    }
}

/// 5x7 глифы для подписи "PERSON #<id>".
fn glyph(c: char) -> Option<[&'static str; 7]> {
    let g = match c {
        'P' => ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."],
        'E' => ["#####", "#....", "#....", "####.", "#....", "#....", "#####"],
        'R' => ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"],
        'S' => [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
        'O' => [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
        'N' => ["#...#", "##..#", "##..#", "#.#.#", "#..##", "#..##", "#...#"],
        '#' => [".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."],
        ' ' => [".....", ".....", ".....", ".....", ".....", ".....", "....."],
        '0' => [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
        '1' => ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
        '2' => [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
        '3' => ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
        '4' => ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
        '5' => ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
        '6' => [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
        '7' => ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
        '8' => [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
        '9' => [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
        _ => return None,
    };
    Some(g)
}
