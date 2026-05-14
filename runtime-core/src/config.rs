//! Чтение и разбор config.json — общий код для backend_gateway и infer_worker.

use std::fs;
use std::path::Path;

use crate::integra::PipelineConfig;

pub fn read_config_json(root: &Path) -> Option<serde_json::Value> {
    let p = root.join("config.json");
    let raw = fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

/// COCO-80 (Ultralytics / YOLO): имя класса → id.
fn coco_class_id_from_label(name: &str) -> Option<i32> {
    const LABELS: &[(&str, i32)] = &[
        ("person", 0), ("bicycle", 1), ("car", 2), ("motorcycle", 3),
        ("airplane", 4), ("bus", 5), ("train", 6), ("truck", 7), ("boat", 8),
        ("traffic light", 9), ("fire hydrant", 10), ("stop sign", 11),
        ("parking meter", 12), ("bench", 13), ("bird", 14), ("cat", 15),
        ("dog", 16), ("horse", 17), ("sheep", 18), ("cow", 19),
        ("elephant", 20), ("bear", 21), ("zebra", 22), ("giraffe", 23),
        ("backpack", 24), ("umbrella", 25), ("handbag", 26), ("tie", 27),
        ("suitcase", 28), ("frisbee", 29), ("skis", 30), ("snowboard", 31),
        ("sports ball", 32), ("kite", 33), ("baseball bat", 34),
        ("baseball glove", 35), ("skateboard", 36), ("surfboard", 37),
        ("tennis racket", 38), ("bottle", 39), ("wine glass", 40),
        ("cup", 41), ("fork", 42), ("knife", 43), ("spoon", 44),
        ("bowl", 45), ("banana", 46), ("apple", 47), ("sandwich", 48),
        ("orange", 49), ("broccoli", 50), ("carrot", 51), ("hot dog", 52),
        ("pizza", 53), ("donut", 54), ("cake", 55), ("chair", 56),
        ("couch", 57), ("potted plant", 58), ("bed", 59), ("dining table", 60),
        ("toilet", 61), ("tv", 62), ("laptop", 63), ("mouse", 64),
        ("remote", 65), ("keyboard", 66), ("cell phone", 67),
        ("microwave", 68), ("oven", 69), ("toaster", 70), ("sink", 71),
        ("refrigerator", 72), ("book", 73), ("clock", 74), ("vase", 75),
        ("scissors", 76), ("teddy bear", 77), ("hair drier", 78),
        ("toothbrush", 79),
    ];
    LABELS.iter().find(|(l, _)| *l == name).map(|(_, id)| *id)
}

fn parse_class_min_conf(model: &serde_json::Value) -> Vec<(i32, f32)> {
    let mut out = Vec::new();
    let Some(obj) = model.get("class_min_conf").and_then(|v| v.as_object()) else {
        return out;
    };
    for (name, val) in obj {
        let Some(id) = coco_class_id_from_label(name.as_str()) else { continue; };
        let Some(th) = val.as_f64() else { continue; };
        if out.len() >= 16 { break; }
        out.push((id, th as f32));
    }
    out
}

pub fn build_pipeline_config(root: &Path, cfg: &serde_json::Value) -> PipelineConfig {
    let mut p = PipelineConfig::default();

    let native = cfg.get("native_analytics");
    let env_engine = std::env::var("INTEGRA_ENGINE_KIND").ok();
    let engine = env_engine
        .or_else(|| {
            native
                .and_then(|n| n.get("engine"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .unwrap_or_else(|| "tensorrt".to_string());
    p.engine_kind = engine;

    let model_rel = native
        .and_then(|n| n.get("model_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("models/yolo11n_fp16.engine");
    p.model_path = root.join(model_rel).to_string_lossy().into_owned();

    p.input_size = native
        .and_then(|n| n.get("input_size"))
        .and_then(|v| v.as_i64())
        .unwrap_or(640) as i32;

    if let Some(model) = cfg.get("model") {
        p.conf_threshold = model.get("conf").and_then(|v| v.as_f64()).unwrap_or(0.25) as f32;
        p.nms_iou_threshold = model.get("iou").and_then(|v| v.as_f64()).unwrap_or(0.45) as f32;
        p.person_class_id = model.get("person_class").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        p.min_box_size_px = model.get("min_box_size_px").and_then(|v| v.as_i64()).unwrap_or(20) as i32;

        if let Some(arr) = model.get("object_classes").and_then(|v| v.as_array()) {
            p.object_classes = arr.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect();
        }

        if model.get("use_native_regional_conf").and_then(|v| v.as_bool()) == Some(true) {
            p.use_regional_class_conf = true;
            p.upper_region_y_ratio = model.get("upper_region_y_ratio").and_then(|v| v.as_f64()).unwrap_or(0.62) as f32;
            p.min_conf_upper = model.get("min_conf_upper").and_then(|v| v.as_f64()).unwrap_or(0.22) as f32;
            p.min_conf_lower = model.get("min_conf_lower").and_then(|v| v.as_f64()).unwrap_or(0.30) as f32;
            p.bottom_region_y_ratio = model.get("bottom_region_y_ratio").and_then(|v| v.as_f64()).unwrap_or(0.88) as f32;
            p.min_conf_bottom = model.get("min_conf_bottom").and_then(|v| v.as_f64()).unwrap_or(0.26) as f32;
            p.border_relax_px = model.get("border_relax_px").and_then(|v| v.as_i64()).unwrap_or(24) as i32;
            p.min_conf_border = model.get("min_conf_border").and_then(|v| v.as_f64()).unwrap_or(0.20) as f32;
            p.person_min_conf_border = model.get("person_min_conf_border").and_then(|v| v.as_f64()).unwrap_or(0.18) as f32;
        }

        p.class_min_conf = parse_class_min_conf(model);
    }

    if let Some(a) = cfg.get("analyzer") {
        macro_rules! set_f64 { ($f:ident, $k:expr) => {
            if let Some(v) = a.get($k).and_then(|v| v.as_f64()) { p.$f = v; }
        }; }
        set_f64!(static_displacement_px, "static_displacement_px");
        set_f64!(static_window_sec, "static_window_sec");
        set_f64!(abandon_time_sec, "abandon_time_sec");
        set_f64!(owner_proximity_px, "owner_proximity_px");
        set_f64!(owner_left_sec, "owner_left_sec");
        set_f64!(disappear_grace_sec, "disappear_grace_sec");
        set_f64!(min_object_area_px, "min_object_area_px");

        if let Some(arr) = a.get("ignore_detection_norm_rect").and_then(|v| v.as_array()) {
            if arr.len() == 4 {
                let nums: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                if nums.len() == 4 && nums[2] > nums[0] && nums[3] > nums[1] {
                    p.ignore_det_norm_x1 = nums[0];
                    p.ignore_det_norm_y1 = nums[1];
                    p.ignore_det_norm_x2 = nums[2];
                    p.ignore_det_norm_y2 = nums[3];
                }
            }
        }

        if let Some(v) = a.get("tracker_iou_match_threshold").and_then(|v| v.as_f64()) {
            p.tracker_iou_match_threshold = v as f32;
        }
        if let Some(v) = a.get("tracker_max_missed_frames").and_then(|v| v.as_i64()) {
            p.tracker_max_missed_frames = v as i32;
        }
        if let Some(v) = a.get("tracker_soft_centroid_match").and_then(|v| v.as_bool()) {
            p.tracker_soft_centroid_match = v;
        }
    }

    p.camera_id = "main".to_string();
    p
}
