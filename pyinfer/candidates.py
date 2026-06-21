"""Class-agnostic объекты сцены (контур M3 по ТЗ).

MOG2 (модель фона) на уменьшенном grayscale → регионы изменения; подавляем
регионы, перекрытые людьми (их ведёт контур M2); назначаем устойчивые ID лёгким
IoU-трекером. Объекту НЕ присваивается класс YOLO (cls_id=-1, cls_name="object").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import cv2
import numpy as np

from .geom import iou_xyxy

WORK_LONG_EDGE = 960
LEARNING_RATE = 0.00035
# Кадры прогрева модели фона после reset.
WARMUP_FRAMES = 60
# Порог бинаризации foreground-маски MOG2. Шум давим темпоральной фильтрацией
# в IoU-трекере (см. ниже), а не задиранием порога.
FG_BIN_THRESH = 140
# Минимальная сторона bbox-региона в координатах ИСХОДНОГО кадра.
# Всё что тоньше/ниже — мусор (артефакты сжатия, тонкие отблески от мрамора).
MIN_REGION_SIDE_PX = 16


@dataclass
class Det:
    bbox: tuple  # (x1,y1,x2,y2) в координатах исходного кадра
    confidence: float = 0.0
    track_id: int = -1
    class_id: int = -1
    cls_name: str = "object"
    owner_near: bool = False  # человек физически перекрывал предмет в жизни трека


class ObjectCandidates:
    """MOG2 → регионы → подавление людей."""

    def __init__(self, min_region_area_px: int = 100):
        self.min_region_area_px = max(1, int(min_region_area_px))
        self._bg = None
        self._reset_model()

    def _reset_model(self):
        # varThreshold повыше — меньше ложного «переднего плана» от бликов/теней
        # на глянцевом полу зала.
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=4000, varThreshold=20.0, detectShadows=False
        )
        self._frames_seen = 0

    def reset(self):
        self._reset_model()

    def process(self, bgr: np.ndarray, persons: list) -> list:
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        long_edge = max(w, h)
        scale = WORK_LONG_EDGE / long_edge if long_edge > WORK_LONG_EDGE else 1.0
        inv = 1.0 / scale if scale > 1e-6 else 1.0

        if scale < 1.0:
            small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self._bg.apply(gray, learningRate=LEARNING_RATE)
        self._frames_seen += 1
        if self._frames_seen <= WARMUP_FRAMES:
            return []  # модель фона ещё прогревается — кандидатов не выдаём
        _, fg = cv2.threshold(fg, FG_BIN_THRESH, 255, cv2.THRESH_BINARY)
        # OPEN большим ядром — давим мелкий шум (артефакты сжатия, дрожь и т.п.).
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        # CLOSE склеивает «дырявые» силуэты неподвижного предмета.
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=3)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area_small = max(4.0, self.min_region_area_px * scale * scale)

        regions = []
        for c in contours:
            if cv2.contourArea(c) < min_area_small:
                continue
            x, y, rw, rh = cv2.boundingRect(c)
            bbox = (x * inv, y * inv, (x + rw) * inv, (y + rh) * inv)
            # минимальная сторона bbox в исходных координатах
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            if bw < MIN_REGION_SIDE_PX or bh < MIN_REGION_SIDE_PX:
                continue
            # фильтр структурности: гладкие блики/тени на полу отсекаем
            if _edge_density(gray, (x, y, x + rw, y + rh)) < EDGE_DENSITY_MIN:
                continue
            # фильтр бликов: яркое + гладкое пятно (низкая std) = солнечный
            # зайчик / отражение. Берём ЦЕНТРАЛЬНЫЕ 60% bbox: после MORPH_CLOSE
            # iterations=3 контур шире пятна на 2-3 пикс, и std по полному bbox
            # перекошен границей с фоном. Центр гарантированно внутри пятна.
            # ВАЖНО — фильтр применяем ТОЛЬКО к мелким регионам: реальные
            # оставленные предметы (белая коробка, лист бумаги, белый рюкзак)
            # под равномерным светом имеют ровно те же mean/std что и блик,
            # но они КРУПНЫЕ. Солнечный зайчик/отражение фары — небольшое пятно.
            # Порог GLINT_MAX_SHORT_SIDE в MOG2-coordinates (small): 22px @960px
            # ≈ 44px в 1920p оригинале.
            short_side = min(rw, rh)
            if short_side <= GLINT_MAX_SHORT_SIDE:
                cx0 = x + rw // 5
                cy0 = y + rh // 5
                cx1 = x + rw - rw // 5
                cy1 = y + rh - rh // 5
                patch = gray[cy0:cy1, cx0:cx1]
                if patch.size > 0:
                    m = float(patch.mean())
                    s = float(patch.std())
                    if m >= GLINT_MEAN_MIN and s <= GLINT_STD_MAX:
                        continue
            if self._on_person(bbox, persons):
                continue
            regions.append(bbox)
        return regions

    @staticmethod
    def _on_person(region, persons) -> bool:
        """Подавляем регион, если он явно «принадлежит» человеку.
        Условия (любое):
          - ≥60% площади региона внутри бокса человека (halo 5%),
          - IoU(person, region) ≥ 0.20.
        Раньше срабатывало по «центроид региона внутри halo» с halo 10%, что
        КРАЙНЕ агрессивно для зала/лавки: коробка на лавке между сидящими
        попадает в halo соседа и дропается до трекера. Удалили это условие и
        ужесточили остальные. Реальная часть тела (рука, нога, тень головы)
        обычно сидит внутри bbox человека на ≥60% — режется. Предмет рядом с
        человеком даже почти вплотную имеет большую часть площади ВНЕ bbox.
        """
        rarea = max(1.0, (region[2] - region[0]) * (region[3] - region[1]))
        for p in persons:
            pb = p.bbox
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            mx, my = 0.05 * pw, 0.05 * ph
            hx1, hy1, hx2, hy2 = pb[0] - mx, pb[1] - my, pb[2] + mx, pb[3] + my
            ix1 = max(region[0], hx1)
            iy1 = max(region[1], hy1)
            ix2 = min(region[2], hx2)
            iy2 = min(region[3], hy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / rarea >= 0.60:
                return True
            if iou_xyxy(pb, region) >= 0.20:
                return True
        return False


@dataclass
class _Track:
    track_id: int
    bbox: tuple
    missed: int = 0
    hits: int = 0          # подряд кадров подтверждения С НЕПОДВИЖНЫМ ЦЕНТРОИДОМ
    stable: bool = False   # достиг порога стабильности
    cx: float = 0.0        # текущий центроид
    cy: float = 0.0
    anchor_cx: float = 0.0 # якорь — точка, относительно которой считаем неподвижность
    anchor_cy: float = 0.0
    ref_patch: object = None  # эталон пикселей в момент стабилизации (np.float32 SZxSZ)
    gone_streak: int = 0      # подряд кадров, когда верификация говорит «предмета нет»
    birth_frame: int = 0      # кадр потока, на котором трек создан
    baseline: bool = False    # фон сцены (есть с самого старта) — НЕ тревожный
    owner_seen: bool = False  # человек перекрывал bbox трека (привязка владельца)
    prev_patch: object = None # патч предыдущего кадра (для межкадрового MAD)
    motion_ema: float = -1.0  # сглаженный межкадровый MAD патча (-1 = ещё не измерен)
    live_streak: int = 0      # подряд кадров «текстура снова живая» (демоция stable)


# Темпоральная фильтрация. Пороги невысокие — от шума защищает пиксельная
# верификация (ниже), а не задранные пороги. Иначе реальный оставленный предмет
# не успевает стать стабильным до того, как FSM мог бы дать тревогу.
CONFIRM_HITS = 4       # кадров неподвижности → трек виден FSM (быстрее ловим предмет)
STABLE_HITS = 12       # кадров неподвижности → трек «стабилен» (≈0.5с @22fps)
# Базовый допуск смещения центроида от якоря. РЕАЛЬНЫЙ допуск адаптивен к размеру
# кадра (см. _eff_static_radius): MOG2 считает на 640px, поэтому дрожание контура
# на 1080p даёт ~10-15px в оригинале и при жёстком радиусе якорь сбрасывался,
# а статичный предмет никогда не набирал STABLE_HITS.
STATIC_RADIUS_PX = 10.0


def _eff_static_radius(gray) -> float:
    if gray is None:
        return STATIC_RADIUS_PX
    h, w = gray.shape[:2]
    diag = math.hypot(w, h)
    return max(STATIC_RADIUS_PX, 0.012 * diag)


# --- Гейт «замершего» предмета (анти-листва / динамическая текстура) ---
# Реальный оставленный предмет ПОСЛЕ постановки перестаёт меняться попиксельно:
# межкадровый MAD патча падает до уровня шума (≈2-8). Листва на ветру, вода, флаги,
# мерцающая тень — «живая текстура», её bbox может стоять на месте (куст не уходит),
# но пиксели внутри меняются всегда → межкадровый MAD остаётся высоким.
# Главная дыра старого кода: регион, который MOG2 видит КАЖДЫЙ кадр (missed==0),
# принимался как «на месте» без пиксельной проверки → листва дозревала до stable и
# давала ложную тревогу. Здесь трек дозревает до stable ТОЛЬКО если сглаженный
# межкадровый MAD ниже порога (IouTracker.frozen_mad). Порог ≤0 → гейт выключен.
MOTION_EMA_ALPHA = 0.35
FROZEN_MAD_DEFAULT = 12.0
# Демоция уже стабильного трека, если его текстура снова «ожила» (motion_ema выше
# порога × этот коэффициент N кадров подряд): листва, успевшая случайно замереть на
# момент стабилизации. Коэффициент с запасом, чтобы не сбрасывать реальный предмет
# при кратком всплеске (смена освещения, проход тени).
DYNAMIC_DEMOTE_FACTOR = 2.5
DYNAMIC_DEMOTE_FRAMES = 20

# --- Анти-«призрак» после сидевшего человека (MOG2 ghost) ---
# Когда человек сидит долго на лавке, MOG2 поглощает его в модель фона. Как только
# встаёт и уходит — настоящий фон под ним отличается от модели → MOG2 выдаёт большой
# регион «переднего плана» в форме того места, где он сидел. Это классический
# ghost-эффект. Frozen-pixel gate его не лечит: фон реально замер.
# Лечим памятью отпечатков: запоминаем bbox каждого почти неподвижного person-трека,
# и когда он исчезает — N кадров блокируем СОЗДАНИЕ новых объект-треков внутри его
# последнего bbox. Если предмет реально оставлен, его трек был создан ещё ПОКА
# человек сидел (MOG2 видит person+thing как одну кляксу) — он уже жив, suppression
# new tracks не трогает существующие. Регион-призрак родится «нуля» → блокируется.
GHOST_SIT_FRAMES = 60        # подряд кадров с почти неподвижным person-bbox → «сидит»
GHOST_SIT_DRIFT_PX = 24.0    # допустимое смещение центроида между кадрами для «сидит»
GHOST_GRACE_FRAMES = 110     # подавляем новые треки в footprint столько кадров (≈5с @22)
GHOST_INFLATE = 0.10         # запас footprint вокруг последнего bbox (10% размера)
GHOST_CENTER_INSIDE = True   # подавляем, если центр НОВОГО региона в footprint

# --- Пиксельная верификация стабильных треков ---
# Когда MOG2 перестаёт видеть стабильный трек, проверяем по эталону, на месте ли
# предмет. Это заменяет «заморозку по таймеру»: реальный оставленный предмет
# живёт пока физически на месте, а фантом (тень/отблеск/остаточный регион)
# исчезает, как только участок кадра вернулся к фону.
PATCH_SZ = 32
PATCH_PRESENT_CORR = 0.45   # corr ≥ → предмет на месте (для текстурных)
PATCH_PRESENT_MAD = 26.0    # MAD ≤ → предмет на месте (работает для ОДНОТОННЫХ)
PATCH_GONE_FRAMES = 30      # столько кадров «нет» подряд → дроп стабильного трека
PERSON_OCCLUDE_IOU = 0.22   # человек рядом с треком — верификацию пропускаем (заморозка)

# Фильтр плотности краёв: реальный предмет имеет контур/структуру, а блик/тень
# на глянцевом полу — гладкие. Регион проходит, только если доля краевых
# пикселей (Canny) в нём ≥ порога. Главный фильтр шума на полу.
EDGE_DENSITY_MIN = 0.040

# Фильтр бликов: солнечный зайчик на мокром асфальте / отражение от стекла —
# яркое, гладкое, низкоконтрастное пятно. Frozen-pixel gate его НЕ режет (блик
# действительно стабилен попиксельно). Признак: средняя яркость патча выше
# GLINT_MEAN_MIN И стандартное отклонение ниже GLINT_STD_MAX. Картон/рюкзак/
# сумка обычно либо темнее, либо имеют выраженную текстуру → проходят.
GLINT_MEAN_MIN = 205.0
GLINT_STD_MAX = 14.0
# Максимальный «короткий бок» региона (в MOG2-small координатах), к которому
# применяем фильтр бликов. Крупнее — это уже не блик, а потенциально предмет.
GLINT_MAX_SHORT_SIDE = 22

# Объект, ставший стабильным в первые BASELINE_FRAMES потока, — фон сцены
# (мебель/начальная обстановка): помечаем baseline=True, в FSM не отдаём.
# Предметы приносят позже (через десятки секунд) → они НЕ baseline → тревожат.
# Защита от абсорбции: если на месте baseline-объекта появляется НОВЫЙ предмет
# (патч меняется сильнее SHED_MAD), метка снимается и предмет снова тревожит.
BASELINE_FRAMES = 330       # ~15с @22fps
BASELINE_SHED_MAD = 48.0    # MAD патча от эталона выше → сцена изменилась → снять baseline

# Привязка владельца. Объект становится виден FSM только дозрев до stable (~0.7с),
# а к этому моменту человек, поставивший предмет, уже отошёл за зону владельца —
# и FSM не успевает зафиксировать owner_near → тревога никогда не срабатывает.
# Поэтому ЗДЕСЬ, на уровне трека региона (живёт с момента появления предмета),
# латчим факт: человек физически перекрыл ≥ OWNER_OVERLAP_FRAC площади предмета
# (поставил / стоял над ним). Этот флаг проносится в FSM как «владелец был».
# Требуем именно перекрытие (а не близость) — толпа рядом не привязывается.
# Дополнительно: владелец должен быть КРУПНЕЕ предмета (человек носит/ставит вещь
# меньше себя). Регион лавки/мебели сопоставим или больше человека → НЕ владелец;
# рюкзак/сумка много меньше → привязывается. Это режет лавочные ложные тревоги.
OWNER_OVERLAP_FRAC = 0.10
OWNER_MAX_OBJ_TO_PERSON = 0.70


def _extract_patch(gray, bbox):
    h, w = gray.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    p = gray[y1:y2, x1:x2]
    p = cv2.resize(p, (PATCH_SZ, PATCH_SZ), interpolation=cv2.INTER_AREA)
    return p.astype(np.float32)


def _patch_present(cur, ref) -> bool:
    """Предмет на месте? Надёжно для ОДНОТОННЫХ объектов (коробка/рюкзак):
    основной критерий — MAD (средняя абс. разница) мал; для текстурных
    дополнительно проходит высокая корреляция.
    """
    if cur is None or ref is None:
        return False
    mad = float(np.abs(cur - ref).mean())
    if mad <= PATCH_PRESENT_MAD:
        return True
    a = cur - cur.mean()
    b = ref - ref.mean()
    da = float(np.sqrt((a * a).sum()))
    db = float(np.sqrt((b * b).sum()))
    if da < 1e-6 or db < 1e-6:
        return False
    return float((a * b).sum() / (da * db)) >= PATCH_PRESENT_CORR


def _edge_density(gray, bbox) -> float:
    """Доля краевых пикселей (Canny) внутри bbox — мера «структурности» региона."""
    h, w = gray.shape[:2]
    x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2])); y2 = min(h, int(bbox[3]))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    roi = gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi, 50, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)


class IouTracker:
    """IoU-трекер регионов: темпоральная фильтрация + пиксельная верификация стабильных треков."""

    def __init__(self, iou_thresh: float = 0.15, max_missed: int = 15,
                 frozen_mad: float = FROZEN_MAD_DEFAULT):
        self.iou_thresh = float(iou_thresh)
        self.max_missed = int(max_missed)
        # Порог межкадрового MAD для дозревания трека до stable (анти-листва).
        # ≤0 → гейт выключен (старое поведение, для A/B сравнения).
        self.frozen_mad = float(frozen_mad)
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._frame = 0  # кадр потока (для baseline-исключения мебели)
        # Память сидящих людей и их «следов» — против MOG2-ghost после ухода.
        # _person_state[tid] = {bbox, cx, cy, sit_frames, last_frame}
        self._person_state: dict[int, dict] = {}
        # _footprints — отпечатки исчезнувших сидевших людей: (bbox, expire_frame).
        self._footprints: list[tuple[tuple, int]] = []

    def reset(self):
        self._tracks = []
        self._next_id = 1
        self._frame = 0
        self._person_state = {}
        self._footprints = []

    def _update_footprints(self, persons):
        """Обновить память сидящих людей; собрать footprints от исчезнувших.

        Логика: person с track_id, чей центроид смещается не более GHOST_SIT_DRIFT_PX
        между кадрами, копит sit_frames. Если он пропал из текущего списка (ушёл) —
        и накопил ≥ GHOST_SIT_FRAMES — пушим его последний bbox в footprints с TTL
        GHOST_GRACE_FRAMES. Они блокируют создание новых объект-треков в этой зоне.
        """
        present = set()
        for p in persons:
            tid = getattr(p, "track_id", -1)
            if tid is None or tid < 0:
                continue  # без id отследить «сидит» не можем
            cx = 0.5 * (p.bbox[0] + p.bbox[2])
            cy = 0.5 * (p.bbox[1] + p.bbox[3])
            st = self._person_state.get(tid)
            if st is None:
                st = {"bbox": tuple(p.bbox), "cx": cx, "cy": cy,
                      "sit_frames": 0, "last_frame": self._frame}
                self._person_state[tid] = st
            else:
                drift = math.hypot(cx - st["cx"], cy - st["cy"])
                if drift <= GHOST_SIT_DRIFT_PX:
                    st["sit_frames"] += 1
                else:
                    st["sit_frames"] = 0
                st["bbox"] = tuple(p.bbox)
                st["cx"], st["cy"] = cx, cy
                st["last_frame"] = self._frame
            present.add(tid)

        # Кто пропал — закрываем footprint, если успел «насидеть» порог.
        gone = [tid for tid in self._person_state if tid not in present]
        for tid in gone:
            st = self._person_state.pop(tid)
            if st["sit_frames"] >= GHOST_SIT_FRAMES:
                bx1, by1, bx2, by2 = st["bbox"]
                bw = (bx2 - bx1) * GHOST_INFLATE
                bh = (by2 - by1) * GHOST_INFLATE
                inflated = (bx1 - bw, by1 - bh, bx2 + bw, by2 + bh)
                self._footprints.append((inflated, self._frame + GHOST_GRACE_FRAMES))

        # Чистим протухшие footprints.
        if self._footprints:
            self._footprints = [(b, exp) for (b, exp) in self._footprints
                                if exp > self._frame]

    def _in_footprint(self, region) -> bool:
        if not self._footprints:
            return False
        rcx = 0.5 * (region[0] + region[2])
        rcy = 0.5 * (region[1] + region[3])
        for (b, _exp) in self._footprints:
            if b[0] <= rcx <= b[2] and b[1] <= rcy <= b[3]:
                return True
        return False

    def update(self, regions: list, persons: list | None = None,
               gray: "np.ndarray | None" = None) -> list:
        persons = persons or []
        self._frame += 1
        # Обновляем footprints ДО матчинга: если новый регион в зоне отпечатка
        # недавно ушедшего сидевшего человека — это MOG2-ghost, не создаём трек.
        self._update_footprints(persons)
        eff_radius = _eff_static_radius(gray)
        used = [False] * len(regions)
        for tr in self._tracks:
            best_iou, best_j = 0.0, -1
            for j, r in enumerate(regions):
                if used[j]:
                    continue
                v = iou_xyxy(tr.bbox, r)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                new_bbox = regions[best_j]
                new_cx = 0.5 * (new_bbox[0] + new_bbox[2])
                new_cy = 0.5 * (new_bbox[1] + new_bbox[3])
                # Дистанция от ЯКОРЯ (первой позиции), не от прошлого кадра, —
                # иначе ползучий шум проходит через всю сцену незаметно.
                if (abs(new_cx - tr.anchor_cx) <= eff_radius
                        and abs(new_cy - tr.anchor_cy) <= eff_radius):
                    tr.hits += 1
                else:
                    tr.hits = 1
                    tr.anchor_cx, tr.anchor_cy = new_cx, new_cy
                # Анти-сжатие bbox: на Linux/другом OpenCV-build MOG2 быстрее
                # поглощает статичный объект в фон → контур ужимается каждый
                # кадр до точки, трек теряется в visualization и patch-verification
                # начинает работать на огрызке. Если новый bbox занимает <50% площади
                # прежнего, держим прежний bbox: пиксельная проверка ниже всё равно
                # удержит трек, если предмет физически на месте. При реальном
                # уменьшении (часть предмета убрали) IoU упадёт и трек уйдёт штатно.
                old_area = max(1.0, (tr.bbox[2] - tr.bbox[0]) * (tr.bbox[3] - tr.bbox[1]))
                new_area = max(1.0, (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1]))
                if new_area < 0.5 * old_area:
                    # держим старый bbox; центроид НЕ перетираем (anchor-логика выше
                    # уже зачла «совпало по центру»), patch-проверка пойдёт по
                    # старому bbox — предмет либо подтвердится, либо умрёт по
                    # PATCH_GONE_FRAMES (нормальный путь).
                    pass
                else:
                    tr.bbox = new_bbox
                tr.cx, tr.cy = new_cx, new_cy
                tr.missed = 0
                tr.gone_streak = 0

                # Межкадровая «живость» региона: для замершего предмета MAD ≈ шум,
                # для листвы/воды/флага — высокий. Сглаживаем EMA. Сравниваем с
                # патчем ПРОШЛОГО кадра (а не с эталоном стабилизации) — это и есть
                # детектор динамической текстуры независимо от того, видит ли MOG2
                # регион каждый кадр.
                if gray is not None:
                    cur_patch = _extract_patch(gray, tr.bbox)
                    if cur_patch is not None:
                        if (tr.prev_patch is not None
                                and tr.prev_patch.shape == cur_patch.shape):
                            inter_mad = float(np.abs(cur_patch - tr.prev_patch).mean())
                            tr.motion_ema = inter_mad if tr.motion_ema < 0 else (
                                MOTION_EMA_ALPHA * inter_mad
                                + (1.0 - MOTION_EMA_ALPHA) * tr.motion_ema)
                        tr.prev_patch = cur_patch

                # Гейт: трек становится stable, только если текстура «замерла».
                # frozen_mad ≤ 0 — гейт выключен. Пока motion_ema не измерен (-1) —
                # не пускаем (нужно хотя бы 2 кадра для межкадрового MAD).
                frozen_ok = (self.frozen_mad <= 0.0) or (
                    0.0 <= tr.motion_ema <= self.frozen_mad)
                if tr.hits >= STABLE_HITS and not tr.stable and frozen_ok:
                    tr.stable = True
                    if gray is not None:
                        tr.ref_patch = _extract_patch(gray, tr.bbox)
                    # стабилизировался в стартовом окне → фон сцены (мебель)
                    if tr.birth_frame <= BASELINE_FRAMES:
                        tr.baseline = True
                elif tr.stable and self.frozen_mad > 0.0 and tr.motion_ema >= 0.0:
                    # Демоция: стабильная текстура снова «ожила» (листва, случайно
                    # замершая на момент стабилизации). N кадров подряд выше порога
                    # × запас → сбрасываем стабильность, трек уходит с экрана/из FSM.
                    if tr.motion_ema > self.frozen_mad * DYNAMIC_DEMOTE_FACTOR:
                        tr.live_streak += 1
                        if tr.live_streak >= DYNAMIC_DEMOTE_FRAMES:
                            tr.stable = False
                            tr.hits = 0
                            tr.ref_patch = None
                            tr.live_streak = 0
                    else:
                        tr.live_streak = 0
                used[best_j] = True
            else:
                # MOG2 не дал контур в этом кадре. Для СТАТИЧНОГО предмета контур
                # мерцает (объект поглощается фоном) — это НЕ движение, поэтому
                # прогресс к стабильности НЕ сбрасываем. hits сбрасывается только
                # при реально зафиксированном движении (match вне радиуса выше).
                tr.missed += 1

        for j, r in enumerate(regions):
            if not used[j]:
                # Анти-ghost: новый регион в свежем footprint сидевшего → дроп.
                # Существующие треки (в т.ч. реальный предмет, оставленный пока
                # человек ещё сидел) уже жили до этого момента и не теряются —
                # они матчатся выше по IoU. Блокируем только РОЖДЕНИЕ из пустоты.
                if self._in_footprint(r):
                    continue
                cx = 0.5 * (r[0] + r[2])
                cy = 0.5 * (r[1] + r[3])
                self._tracks.append(_Track(
                    self._next_id, r, hits=1,
                    cx=cx, cy=cy, anchor_cx=cx, anchor_cy=cy,
                    birth_frame=self._frame,
                ))
                self._next_id += 1

        # --- Привязка владельца: латчим перекрытие предмета человеком ---
        # Делаем для ВСЕХ треков (включая ещё не stable) — именно в ранние кадры,
        # пока хозяин не отошёл, есть перекрытие bbox.
        if persons:
            for tr in self._tracks:
                if tr.owner_seen:
                    continue
                rarea = max(1.0, (tr.bbox[2] - tr.bbox[0]) * (tr.bbox[3] - tr.bbox[1]))
                for p in persons:
                    pb = p.bbox
                    parea = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
                    if rarea > OWNER_MAX_OBJ_TO_PERSON * parea:
                        continue  # предмет крупнее носителя → это не оставленная вещь
                    ix = max(0.0, min(tr.bbox[2], pb[2]) - max(tr.bbox[0], pb[0]))
                    iy = max(0.0, min(tr.bbox[3], pb[3]) - max(tr.bbox[1], pb[1]))
                    if (ix * iy) / rarea >= OWNER_OVERLAP_FRAC:
                        tr.owner_seen = True
                        break

        # --- Верификация стабильных треков, не подтверждённых MOG2 в этом кадре ---
        survivors = []
        for tr in self._tracks:
            # Снятие baseline: на месте мебели появился НОВЫЙ предмет (патч сильно
            # изменился) → это уже не фон, делаем трек тревожным и обновляем эталон.
            if tr.baseline and gray is not None and tr.ref_patch is not None:
                cur = _extract_patch(gray, tr.bbox)
                if cur is not None and float(np.abs(cur - tr.ref_patch).mean()) > BASELINE_SHED_MAD:
                    tr.baseline = False
                    tr.ref_patch = cur

            if not tr.stable:
                # обычный трек: живёт пока missed <= max_missed
                if tr.missed <= self.max_missed:
                    survivors.append(tr)
                continue

            if tr.missed == 0:
                survivors.append(tr)  # подтверждён движением — точно на месте
                continue

            occluded = any(iou_xyxy(tr.bbox, p.bbox) >= PERSON_OCCLUDE_IOU for p in persons)
            if occluded:
                # человек закрывает предмет — не можем проверить, считаем «на месте»
                tr.gone_streak = 0
                survivors.append(tr)
                continue

            if gray is None or tr.ref_patch is None:
                # без кадра — старое поведение: ограниченная заморозка
                if tr.missed <= self.max_missed * 4:
                    survivors.append(tr)
                continue

            if _patch_present(_extract_patch(gray, tr.bbox), tr.ref_patch):
                tr.gone_streak = 0
                survivors.append(tr)  # предмет физически на месте → держим
            else:
                tr.gone_streak += 1
                if tr.gone_streak < PATCH_GONE_FRAMES:
                    survivors.append(tr)
                # иначе дропаем: участок вернулся к фону → предмет исчез/забрали
        self._tracks = survivors

        # FSM/оверлею отдаём ТОЛЬКО стабильные (пиксельно-верифицированные) треки.
        # «Подтверждённые» (CONFIRM..STABLE) — внутренняя стадия дозревания, на
        # экран не выходят: так транзиентный шум (тени/отблески/мелькание) не
        # порождает жёлтых рамок, а реальный предмет показывается, дозрев до stable.
        out = []
        for t in self._tracks:
            if not t.stable or t.baseline:
                continue  # baseline = фон сцены (мебель) — в FSM не выходит
            out.append(Det(bbox=t.bbox, confidence=0.5, track_id=t.track_id,
                           owner_near=t.owner_seen))
        return out
