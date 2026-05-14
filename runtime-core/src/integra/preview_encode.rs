//! JPEG превью с bbox-overlay и кроп снапшотов для тревог.

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};

use super::events::{PersonDet, TrackSnapshot};

/// Длинная сторона превью перед JPEG (выше — лучше качество, больше CPU/память на кадр).
const PREVIEW_ENCODE_MAX_EDGE: u32 = 1600;
const PREVIEW_JPEG_QUALITY: u8 = 86;

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
    if w == 0 || h == 0 || bgr.len() < (w as usize) * (h as usize) * 3 {
        return None;
    }

    let scale = if w.max(h) > PREVIEW_ENCODE_MAX_EDGE {
        PREVIEW_ENCODE_MAX_EDGE as f32 / w.max(h) as f32
    } else {
        1.0f32
    };
    let ow = (w as f32 * scale).max(1.0).round() as u32;
    let oh = (h as f32 * scale).max(1.0).round() as u32;
    let inv = 1.0f32 / scale;

    let mut rgb = vec![0u8; (ow * oh * 3) as usize];
    for ty in 0..oh {
        let src_y_f = (ty as f32 + 0.5) * inv - 0.5;
        let src_y = src_y_f.clamp(0.0, h as f32 - 1.0).round() as u32;
        for tx in 0..ow {
            let src_x_f = (tx as f32 + 0.5) * inv - 0.5;
            let src_x = src_x_f.clamp(0.0, w as f32 - 1.0).round() as u32;
            let si = ((src_y * w + src_x) * 3) as usize;
            let di = ((ty * ow + tx) * 3) as usize;
            if si + 2 < bgr.len() && di + 2 < rgb.len() {
                rgb[di] = bgr[si + 2];
                rgb[di + 1] = bgr[si + 1];
                rgb[di + 2] = bgr[si];
            }
        }
    }

    let iw = ow as i32;
    let ih = oh as i32;
    for t in tracks {
        if !track_visible_on_mjpeg_preview(t.state.as_str()) {
            continue;
        }
        let color = color_for_state(&t.state);
        let thickness = 2;
        let x1 = (t.bbox[0].min(t.bbox[2]) * scale) as i32;
        let y1 = (t.bbox[1].min(t.bbox[3]) * scale) as i32;
        let x2 = (t.bbox[0].max(t.bbox[2]) * scale) as i32;
        let y2 = (t.bbox[1].max(t.bbox[3]) * scale) as i32;
        draw_rect(&mut rgb, iw, ih, x1, y1, x2, y2, color, thickness);
    }
    // Люди поверх объектов (пурпурный — не путать с бирюзовым static и не «сигналить» зелёным).
    if draw_person_boxes {
        let person_color = [230, 60, 255];
        for p in persons {
            let x1 = (p.bbox[0].min(p.bbox[2]) * scale) as i32;
            let y1 = (p.bbox[1].min(p.bbox[3]) * scale) as i32;
            let x2 = (p.bbox[0].max(p.bbox[2]) * scale) as i32;
            let y2 = (p.bbox[1].max(p.bbox[3]) * scale) as i32;
            draw_rect(&mut rgb, iw, ih, x1, y1, x2, y2, person_color, 3);
        }
    }

    let img = ImageBuffer::<Rgb<u8>, _>::from_raw(ow, oh, rgb)?;
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, PREVIEW_JPEG_QUALITY);
    enc.encode_image(&img).ok()?;
    Some(jpeg)
}

/// На превью не показываем «тихие» состояния (`static` / `candidate`).
fn track_visible_on_mjpeg_preview(state: &str) -> bool {
    matches!(
        state,
        "unattended" | "alarm_abandoned" | "alarm_disappeared"
    )
}

fn color_for_state(state: &str) -> [u8; 3] {
    match state {
        "candidate" => [255, 200, 0],
        "static" => [95, 130, 140],
        "unattended" => [255, 140, 0],
        "alarm_abandoned" => [255, 0, 0],
        "alarm_disappeared" => [255, 0, 200],
        _ => [180, 180, 180],
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
