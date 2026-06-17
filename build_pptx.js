const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
pres.author = "Detection / Integra";
pres.title = "Отчёт. Детектор оставленных предметов";

// Palette: Midnight Executive
const NAVY = "1E2761";
const NAVY_DEEP = "141A45";
const ICE = "CADCFC";
const WHITE = "FFFFFF";
const ACCENT = "F0B429";
const MUTED = "8A93B8";
const CARD = "F4F7FE";
const TEXT_DARK = "1A1F3A";
const GREEN = "27C4A4";
const RED = "E55353";

const W = 13.3, H = 7.5;
const HEADER_FONT = "Calibri";
const BODY_FONT = "Calibri";

const TOTAL = 9;

function addContentHeader(slide, title, kicker) {
  slide.background = { color: WHITE };
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.9, fill: { color: NAVY }, line: { color: NAVY }
  });
  slide.addText(kicker, {
    x: 0.6, y: 0.18, w: 8, h: 0.3,
    fontSize: 11, fontFace: HEADER_FONT, color: ICE, bold: true, charSpacing: 4, margin: 0
  });
  slide.addText(title, {
    x: 0.6, y: 0.4, w: 12, h: 0.5,
    fontSize: 24, fontFace: HEADER_FONT, color: WHITE, bold: true, margin: 0
  });
  slide.addShape(pres.shapes.OVAL, {
    x: W - 0.55, y: 0.38, w: 0.18, h: 0.18, fill: { color: ACCENT }, line: { color: ACCENT }
  });
}

function addFooter(slide, page) {
  slide.addText(`Интегра  •  Отчёт о проделанной работе`, {
    x: 0.6, y: H - 0.4, w: 8, h: 0.3,
    fontSize: 9, fontFace: BODY_FONT, color: MUTED, margin: 0
  });
  slide.addText(`${page} / ${TOTAL}`, {
    x: W - 1.4, y: H - 0.4, w: 0.8, h: 0.3,
    fontSize: 9, fontFace: BODY_FONT, color: MUTED, align: "right", margin: 0
  });
}

// =============================================================
// Slide 1 — Title
// =============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY_DEEP };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.25, h: H, fill: { color: ACCENT }, line: { color: ACCENT }
  });

  s.addText("ОТЧЁТ О ПРОДЕЛАННОЙ РАБОТЕ  •  2026", {
    x: 0.8, y: 1.3, w: 10, h: 0.4,
    fontSize: 12, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 8, margin: 0
  });

  s.addText("Детектор оставленных\nпредметов", {
    x: 0.8, y: 1.8, w: 12, h: 2.4,
    fontSize: 48, fontFace: HEADER_FONT, color: WHITE, bold: true, margin: 0
  });

  s.addText("Альтернатива традиционной видеоаналитике\nна основе анализа с помощью искусственного интеллекта", {
    x: 0.8, y: 4.4, w: 11, h: 1.2,
    fontSize: 20, fontFace: BODY_FONT, color: ICE, margin: 0
  });

  s.addShape(pres.shapes.LINE, {
    x: 0.8, y: 6.0, w: 5.0, h: 0,
    line: { color: ACCENT, width: 2 }
  });
  s.addText("Стенд системы видеонаблюдения  •  офис Интегры", {
    x: 0.8, y: 6.15, w: 9, h: 0.4,
    fontSize: 13, fontFace: BODY_FONT, color: ICE, margin: 0
  });
  s.addText("Направление аналитики ВПП  •  Мартынов", {
    x: 0.8, y: 6.55, w: 9, h: 0.4,
    fontSize: 11, fontFace: BODY_FONT, color: MUTED, italic: true, margin: 0
  });
}

// =============================================================
// Slide 2 — Идея проекта
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Идея проекта", "01 / ИДЕЯ");

  // big quote / idea on left
  s.addText("Альтернатива традиционной видеоаналитике\nоставленного предмета", {
    x: 0.6, y: 1.3, w: 7.2, h: 1.5,
    fontSize: 22, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });

  s.addShape(pres.shapes.LINE, {
    x: 0.6, y: 2.9, w: 1.5, h: 0, line: { color: ACCENT, width: 3 }
  });

  s.addText("В основе подхода — анализ изображения с помощью искусственного интеллекта (нейросети) вместо классических алгоритмов на правилах и фоновом вычитании.", {
    x: 0.6, y: 3.1, w: 7.0, h: 1.5,
    fontSize: 14, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
  });

  s.addText("Зачем", {
    x: 0.6, y: 4.85, w: 7.0, h: 0.4,
    fontSize: 14, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText([
    { text: "ИИ устойчивее к смене освещения, теням, людям в кадре", options: { bullet: true, breakLine: true } },
    { text: "Понимает, что именно оставлено — сумка, чемодан, рюкзак", options: { bullet: true, breakLine: true } },
    { text: "Различает владельца и объект, оценивает их связь", options: { bullet: true } },
  ], {
    x: 0.6, y: 5.3, w: 7.0, h: 1.7,
    fontSize: 13, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, margin: 0
  });

  // right column — visual contrast
  const baseX = 8.0, cardW = 4.7;
  // Traditional
  s.addShape(pres.shapes.RECTANGLE, {
    x: baseX, y: 1.3, w: cardW, h: 2.7, fill: { color: CARD }, line: { color: CARD }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: baseX, y: 1.3, w: 0.08, h: 2.7, fill: { color: RED }, line: { color: RED }
  });
  s.addText("Традиционный подход", {
    x: baseX + 0.25, y: 1.45, w: cardW - 0.4, h: 0.4,
    fontSize: 14, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText([
    { text: "Правила и фоновое вычитание", options: { bullet: true, breakLine: true } },
    { text: "Чувствителен к свету и теням", options: { bullet: true, breakLine: true } },
    { text: "Не отличает сумку от стула", options: { bullet: true, breakLine: true } },
    { text: "Много ложных срабатываний", options: { bullet: true } },
  ], {
    x: baseX + 0.25, y: 1.9, w: cardW - 0.4, h: 2.0,
    fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, margin: 0
  });

  // Our approach
  s.addShape(pres.shapes.RECTANGLE, {
    x: baseX, y: 4.15, w: cardW, h: 2.7, fill: { color: CARD }, line: { color: CARD }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: baseX, y: 4.15, w: 0.08, h: 2.7, fill: { color: GREEN }, line: { color: GREEN }
  });
  s.addText("Наш подход", {
    x: baseX + 0.25, y: 4.3, w: cardW - 0.4, h: 0.4,
    fontSize: 14, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText([
    { text: "Распознавание объектов на ИИ", options: { bullet: true, breakLine: true } },
    { text: "Знает, что это сумка / чемодан", options: { bullet: true, breakLine: true } },
    { text: "Связывает предмет и владельца", options: { bullet: true, breakLine: true } },
    { text: "Ложных тревог в разы меньше", options: { bullet: true } },
  ], {
    x: baseX + 0.25, y: 4.75, w: cardW - 0.4, h: 2.0,
    fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, margin: 0
  });

  addFooter(s, 2);
}

// =============================================================
// Slide 3 — Цели проекта
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Цели проекта", "02 / ЦЕЛИ");

  const goals = [
    {
      n: "01",
      t: "Повышение качества",
      d: "Меньше ложных тревог. Стабильное обнаружение оставленных сумок, чемоданов и рюкзаков в реальных условиях видеонаблюдения.",
      c: ACCENT,
    },
    {
      n: "02",
      t: "Снижение ресурсоёмкости",
      d: "Меньше нагрузки на сервер видеоаналитики: видеопамять, оперативная память, центральный процессор. Больше камер на одном сервере.",
      c: GREEN,
    },
    {
      n: "03",
      t: "Гибкость и развитие",
      d: "Архитектура, в которой видеоаналитика отделена от оценки событий. Это упрощает развитие, замену моделей и подключение новых детекторов.",
      c: "4F9DDE",
    },
  ];

  const baseY = 1.3, cardH = 1.7, gap = 0.25;
  goals.forEach((g, i) => {
    const y = baseY + i * (cardH + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: y, w: 12.1, h: cardH, fill: { color: CARD }, line: { color: CARD }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6, y: y, w: 0.12, h: cardH, fill: { color: g.c }, line: { color: g.c }
    });
    s.addText(g.n, {
      x: 0.95, y: y + 0.3, w: 1.4, h: 1.1,
      fontSize: 48, fontFace: HEADER_FONT, color: g.c, bold: true, margin: 0
    });
    s.addText(g.t, {
      x: 2.55, y: y + 0.25, w: 9.8, h: 0.6,
      fontSize: 20, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
    });
    s.addText(g.d, {
      x: 2.55, y: y + 0.85, w: 10.0, h: 0.8,
      fontSize: 13, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
    });
  });

  addFooter(s, 3);
}

// =============================================================
// Slide 4 — Что сделано: работы
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Что сделано", "03 / РАБОТЫ");

  // Timeline-like 3 columns
  const cols = [
    {
      t: "1.  Основа подхода",
      h: "Анализ от ИИ",
      d: "За основу взят анализ изображения с помощью нейросетевой модели распознавания объектов. Это даёт качественно другой уровень понимания сцены по сравнению с классическими алгоритмами.",
    },
    {
      t: "2.  Стенд для работ",
      h: "Офис Интегры",
      d: "Использован стенд системы видеонаблюдения в офисе Интегры (направление аналитики ВПП — Мартынов). Реальные камеры, реальная сеть, реальные условия.",
    },
    {
      t: "3.  Данные для обучения\n     и оценки",
      h: "Видео из фойе офиса",
      d: "Для обучения и оценки качества используется видео с камер фойе офиса. Это типовая сцена под целевую задачу: проходящие люди и оставленные вещи.",
    },
  ];

  const baseX = 0.6, baseY = 1.3, cardW = 4.07, cardH = 5.6, gap = 0.13;
  cols.forEach((c, i) => {
    const x = baseX + i * (cardW + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: baseY, w: cardW, h: cardH, fill: { color: WHITE }, line: { color: ICE, width: 1 }
    });
    // step header
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: baseY, w: cardW, h: 0.7, fill: { color: NAVY }, line: { color: NAVY }
    });
    s.addText(c.t, {
      x: x + 0.25, y: baseY + 0.05, w: cardW - 0.4, h: 0.65,
      fontSize: 12, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 4, valign: "middle", margin: 0
    });
    // big heading
    s.addText(c.h, {
      x: x + 0.3, y: baseY + 0.9, w: cardW - 0.5, h: 1.0,
      fontSize: 22, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
    });
    // body
    s.addText(c.d, {
      x: x + 0.3, y: baseY + 2.1, w: cardW - 0.5, h: 3.3,
      fontSize: 13, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
    });
  });

  addFooter(s, 4);
}

// =============================================================
// Slide 5 — Новый подход (архитектура простыми словами)
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Новый подход", "04 / ПОДХОД");

  s.addText("Видеоаналитика и оценка событий — два разных процесса", {
    x: 0.6, y: 1.15, w: 12, h: 0.5,
    fontSize: 16, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });

  // Stage 1 box
  const sx1 = 0.6, sy = 1.85, sw = 5.8, sh = 4.4;
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx1, y: sy, w: sw, h: sh, fill: { color: CARD }, line: { color: CARD }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx1, y: sy, w: sw, h: 0.6, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("ШАГ 1.  Анализ кадров", {
    x: sx1 + 0.25, y: sy, w: sw - 0.4, h: 0.6,
    fontSize: 13, fontFace: HEADER_FONT, color: WHITE, bold: true, charSpacing: 4, valign: "middle", margin: 0
  });

  s.addText("Что происходит", {
    x: sx1 + 0.3, y: sy + 0.8, w: sw - 0.6, h: 0.4,
    fontSize: 13, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText("Система смотрит каждый кадр и последовательность кадров. Распознаёт предметы и людей, отслеживает их во времени.", {
    x: sx1 + 0.3, y: sy + 1.15, w: sw - 0.6, h: 1.2,
    fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
  });

  s.addText("Что на выходе", {
    x: sx1 + 0.3, y: sy + 2.4, w: sw - 0.6, h: 0.4,
    fontSize: 13, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText([
    { text: "Камера, на которой сработало", options: { bullet: true, breakLine: true } },
    { text: "Зона в кадре и координаты предмета", options: { bullet: true, breakLine: true } },
    { text: "Точное время события", options: { bullet: true, breakLine: true } },
    { text: "Снимок-доказательство", options: { bullet: true } },
  ], {
    x: sx1 + 0.3, y: sy + 2.75, w: sw - 0.6, h: 1.6,
    fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, margin: 0
  });

  // arrow between
  s.addShape(pres.shapes.RIGHT_TRIANGLE, {
    x: 6.55, y: sy + sh / 2 - 0.25, w: 0.5, h: 0.5,
    fill: { color: ACCENT }, line: { color: ACCENT }, rotate: 90,
  });
  s.addText("событие", {
    x: 6.3, y: sy + sh / 2 + 0.3, w: 1.1, h: 0.3,
    fontSize: 10, fontFace: BODY_FONT, color: NAVY, italic: true, align: "center", margin: 0
  });

  // Stage 2 box
  const sx2 = 7.2;
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx2, y: sy, w: sw, h: sh, fill: { color: CARD }, line: { color: CARD }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: sx2, y: sy, w: sw, h: 0.6, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("ШАГ 2.  Оценка событий", {
    x: sx2 + 0.25, y: sy, w: sw - 0.4, h: 0.6,
    fontSize: 13, fontFace: HEADER_FONT, color: WHITE, bold: true, charSpacing: 4, valign: "middle", margin: 0
  });

  s.addText("Что происходит", {
    x: sx2 + 0.3, y: sy + 0.8, w: sw - 0.6, h: 0.4,
    fontSize: 13, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText("События обрабатывает отдельный процесс. Решает, какие события важны, как их группировать, кому показывать.", {
    x: sx2 + 0.3, y: sy + 1.15, w: sw - 0.6, h: 1.2,
    fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
  });

  s.addText("Что даёт разделение", {
    x: sx2 + 0.3, y: sy + 2.4, w: sw - 0.6, h: 0.4,
    fontSize: 13, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText([
    { text: "Гибкая логика тревог под объект", options: { bullet: true, breakLine: true } },
    { text: "Возможность подключить ИИ-оценку", options: { bullet: true, breakLine: true } },
    { text: "Корреляция событий разных камер", options: { bullet: true, breakLine: true } },
    { text: "Развитие без переделки видеоаналитики", options: { bullet: true } },
  ], {
    x: sx2 + 0.3, y: sy + 2.75, w: sw - 0.6, h: 1.6,
    fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, margin: 0
  });

  addFooter(s, 5);
}

// =============================================================
// Slide 6 — Результат: качество
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Результат  •  качество", "05 / РЕЗУЛЬТАТ");

  // Top: before/after-style two big numbers
  const bigCardY = 1.25, bigCardH = 1.85;
  // Before
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: bigCardY, w: 6.05, h: bigCardH, fill: { color: CARD }, line: { color: CARD }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: bigCardY, w: 0.12, h: bigCardH, fill: { color: RED }, line: { color: RED }
  });
  s.addText("ДО ДОРАБОТОК", {
    x: 0.9, y: bigCardY + 0.2, w: 5.5, h: 0.35,
    fontSize: 11, fontFace: HEADER_FONT, color: RED, bold: true, charSpacing: 6, margin: 0
  });
  s.addText("Частые ложные тревоги", {
    x: 0.9, y: bigCardY + 0.6, w: 5.6, h: 0.6,
    fontSize: 22, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText("Сломанные временны́е метки, низкие пороги уверенности, рассогласованное слежение, мерцающие мелкие объекты.", {
    x: 0.9, y: bigCardY + 1.2, w: 5.6, h: 0.7,
    fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
  });

  // After
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.85, y: bigCardY, w: 6.05, h: bigCardH, fill: { color: CARD }, line: { color: CARD }
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.85, y: bigCardY, w: 0.12, h: bigCardH, fill: { color: GREEN }, line: { color: GREEN }
  });
  s.addText("ПОСЛЕ ДОРАБОТОК", {
    x: 7.15, y: bigCardY + 0.2, w: 5.5, h: 0.35,
    fontSize: 11, fontFace: HEADER_FONT, color: GREEN, bold: true, charSpacing: 6, margin: 0
  });
  s.addText("Стабильное обнаружение", {
    x: 7.15, y: bigCardY + 0.6, w: 5.6, h: 0.6,
    fontSize: 22, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });
  s.addText("Корректное время событий, согласованные пороги детектора и слежения, отсечены мелкие шумовые объекты.", {
    x: 7.15, y: bigCardY + 1.2, w: 5.6, h: 0.7,
    fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
  });

  // List of fixes
  const fixes = [
    { t: "Исправлена передача времени", d: "Временны́е метки кадров перестали приходить нулевыми — логика тревог по времени работает корректно." },
    { t: "Согласованы пороги уверенности", d: "Подобраны под реальные условия камер. Меньше ложных тревог, реальные сумки и чемоданы по-прежнему ловятся." },
    { t: "Согласовано слежение за объектами", d: "Параметры алгоритма слежения подогнаны к настройкам распознавания — пропали «фантомные» объекты." },
    { t: "Отсечены мелкие шумовые объекты", d: "Поднят минимальный размер обнаруженного предмета, увеличено время удержания трека." },
  ];

  const lY = bigCardY + bigCardH + 0.3;
  s.addText("Ключевые улучшения", {
    x: 0.6, y: lY, w: 12, h: 0.4,
    fontSize: 14, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });

  fixes.forEach((f, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 6.25;
    const y = lY + 0.55 + row * 0.95;
    s.addShape(pres.shapes.OVAL, {
      x: x, y: y + 0.05, w: 0.25, h: 0.25, fill: { color: ACCENT }, line: { color: ACCENT }
    });
    s.addText(f.t, {
      x: x + 0.4, y: y, w: 5.7, h: 0.35,
      fontSize: 12, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
    });
    s.addText(f.d, {
      x: x + 0.4, y: y + 0.32, w: 5.7, h: 0.6,
      fontSize: 10, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0
    });
  });

  addFooter(s, 6);
}

// =============================================================
// Slide 7 — Результат: ресурсы
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Результат  •  ресурсы", "06 / РЕСУРСЫ");

  // Headline metric
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.6, y: 1.2, w: 12.1, h: 2.0, fill: { color: NAVY }, line: { color: NAVY }
  });
  s.addText("ОРИЕНТИРОВОЧНАЯ НАГРУЗКА  •  10 КАМЕР  •  2 МП", {
    x: 0.9, y: 1.4, w: 11.5, h: 0.4,
    fontSize: 12, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 6, margin: 0
  });

  const metrics = [
    { v: "~ 3 ГБ", l: "Оперативная\nпамять (ОЗУ)" },
    { v: "~ 2 ГБ", l: "Видеопамять\nна одной видеокарте" },
    { v: "~ 35 %", l: "Загрузка\nцентрального процессора" },
    { v: "1 сервер", l: "Достаточно одного\nсервера с видеокартой" },
  ];
  const mx0 = 0.9, mw = 2.95, mgap = 0.05, my = 1.9;
  metrics.forEach((m, i) => {
    const x = mx0 + i * (mw + mgap);
    s.addText(m.v, {
      x: x, y: my, w: mw, h: 0.7,
      fontSize: 32, fontFace: HEADER_FONT, color: WHITE, bold: true, margin: 0
    });
    s.addText(m.l, {
      x: x, y: my + 0.75, w: mw, h: 0.6,
      fontSize: 11, fontFace: BODY_FONT, color: ICE, margin: 0
    });
  });

  s.addText("Оценка по итогам стенда; уточняется на типовом профиле нагрузки заказчика.", {
    x: 0.6, y: 3.3, w: 12, h: 0.3,
    fontSize: 10, fontFace: BODY_FONT, color: MUTED, italic: true, margin: 0
  });

  // Comparison table
  s.addText("Сравнение с традиционным подходом", {
    x: 0.6, y: 3.75, w: 12, h: 0.4,
    fontSize: 14, fontFace: HEADER_FONT, color: NAVY, bold: true, margin: 0
  });

  const tHead = (txt) => ({ text: txt, options: { fill: { color: NAVY }, color: WHITE, bold: true, align: "left", valign: "middle", fontSize: 12, fontFace: HEADER_FONT } });
  const tCell = (txt, c = TEXT_DARK) => ({ text: txt, options: { color: c, fontSize: 12, fontFace: BODY_FONT, valign: "middle", align: "left" } });

  const tableData = [
    [tHead("  Показатель"), tHead("  Традиционный подход"), tHead("  Наш подход")],
    [tCell("  Качество в реальных условиях"), tCell("  Чувствительно к свету, теням, людям"), tCell("  Стабильно, объект понимается семантически", GREEN)],
    [tCell("  Видеопамять"), tCell("  Часто по экземпляру модели на поток"), tCell("  Один общий движок на все камеры", GREEN)],
    [tCell("  Оперативная память"), tCell("  Копирование кадров между этапами"), tCell("  Передача кадров по ссылке без копий", GREEN)],
    [tCell("  Развитие и интеграция"), tCell("  Логика вшита в детектор"), tCell("  Аналитика и оценка событий разделены", GREEN)],
  ];

  s.addTable(tableData, {
    x: 0.6, y: 4.25, w: 12.1,
    colW: [3.6, 4.3, 4.2],
    rowH: 0.45,
    border: { pt: 1, color: ICE },
  });

  addFooter(s, 7);
}

// =============================================================
// Slide 8 — Гибкость настройки в схеме ситуационной аналитики
// =============================================================
{
  const s = pres.addSlide();
  addContentHeader(s, "Гибкость в схеме ситуационной аналитики", "07 / НАСТРОЙКА");

  s.addText("Под каждый объект и сценарий — свои настройки без пересборки системы", {
    x: 0.6, y: 1.15, w: 12, h: 0.4,
    fontSize: 14, fontFace: HEADER_FONT, color: NAVY, italic: true, margin: 0
  });

  const groups = [
    {
      t: "Что искать и где",
      items: [
        "Список предметов: что интересно, что игнорируем",
        "Зоны интереса в кадре и зоны игнорирования",
        "Учёт особенностей сцены (стенды, экраны, окна)",
      ],
    },
    {
      t: "Когда поднимать тревогу",
      items: [
        "Сколько времени предмет должен быть оставлен",
        "На каком расстоянии искать «владельца»",
        "Сколько владелец должен отсутствовать рядом",
      ],
    },
    {
      t: "Учёт особенностей камеры",
      items: [
        "Разные пороги для разных зон кадра",
        "Учёт перспективы и краёв изображения",
        "Минимальный размер объекта по типам",
      ],
    },
    {
      t: "Куда уходят события",
      items: [
        "Подписка на события в реальном времени",
        "Видео с разметкой найденных объектов",
        "Снимок-доказательство для каждой тревоги",
      ],
    },
  ];

  const cols = 2, baseX = 0.6, baseY = 1.7, cardW = 6.05, cardH = 2.55, gap = 0.2;
  groups.forEach((g, i) => {
    const col = i % cols, row = Math.floor(i / cols);
    const x = baseX + col * (cardW + gap);
    const y = baseY + row * (cardH + gap);
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: cardW, h: cardH, fill: { color: WHITE }, line: { color: ICE, width: 1 }
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: cardW, h: 0.5, fill: { color: NAVY }, line: { color: NAVY }
    });
    s.addText(g.t, {
      x: x + 0.25, y: y, w: cardW - 0.4, h: 0.5,
      fontSize: 14, fontFace: HEADER_FONT, color: WHITE, bold: true, valign: "middle", margin: 0
    });
    s.addText(
      g.items.map((it, j) => ({
        text: it, options: { bullet: true, breakLine: j < g.items.length - 1 }
      })),
      {
        x: x + 0.3, y: y + 0.65, w: cardW - 0.5, h: cardH - 0.8,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 6, margin: 0
      }
    );
  });

  addFooter(s, 8);
}

// =============================================================
// Slide 9 — Вывод
// =============================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY_DEEP };

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.25, h: H, fill: { color: ACCENT }, line: { color: ACCENT }
  });

  s.addText("ВЫВОД", {
    x: 0.8, y: 0.45, w: 6, h: 0.4,
    fontSize: 12, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 8, margin: 0
  });
  s.addText("Цели достигнуты, заложен новый подход", {
    x: 0.8, y: 0.85, w: 12, h: 1.0,
    fontSize: 28, fontFace: HEADER_FONT, color: WHITE, bold: true, margin: 0
  });

  // 3 columns: goals achieved / new approach / future
  const cY = 2.15, cH = 4.5;

  // Col 1: goals achieved
  s.addText("ЦЕЛИ ПРОЕКТА", {
    x: 0.8, y: cY, w: 4.0, h: 0.35,
    fontSize: 10, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 6, margin: 0
  });
  const achievements = [
    { k: "Качество", v: "Меньше ложных тревог, стабильное обнаружение сумок и чемоданов на стенде." },
    { k: "Ресурсы", v: "~3 ГБ ОЗУ и ~2 ГБ видеопамяти на 10 камер 2 МП — один сервер вместо нескольких." },
    { k: "Гибкость", v: "Аналитика и оценка событий разделены — упрощено развитие и интеграция." },
  ];
  let ay = cY + 0.5;
  achievements.forEach(a => {
    s.addText(a.k, {
      x: 0.8, y: ay, w: 4.0, h: 0.35,
      fontSize: 13, fontFace: HEADER_FONT, color: WHITE, bold: true, margin: 0
    });
    s.addText(a.v, {
      x: 0.8, y: ay + 0.35, w: 4.0, h: 1.0,
      fontSize: 11, fontFace: BODY_FONT, color: ICE, margin: 0
    });
    ay += 1.3;
  });

  // Col 2: new approach
  const c2x = 5.0;
  s.addText("ГЛАВНОЕ  •  НОВЫЙ ПОДХОД", {
    x: c2x, y: cY, w: 4.0, h: 0.35,
    fontSize: 10, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 6, margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: c2x, y: cY + 0.5, w: 4.0, h: 1.85,
    fill: { color: NAVY }, line: { color: ACCENT, width: 1 }
  });
  s.addText("Шаг 1.  Анализ кадров", {
    x: c2x + 0.2, y: cY + 0.6, w: 3.7, h: 0.35,
    fontSize: 12, fontFace: HEADER_FONT, color: ACCENT, bold: true, margin: 0
  });
  s.addText("Анализ кадра и последовательности кадров → событие: камера, зона / координаты, время.", {
    x: c2x + 0.2, y: cY + 0.95, w: 3.7, h: 1.3,
    fontSize: 11, fontFace: BODY_FONT, color: ICE, margin: 0
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: c2x, y: cY + 2.5, w: 4.0, h: 1.85,
    fill: { color: NAVY }, line: { color: ACCENT, width: 1 }
  });
  s.addText("Шаг 2.  Оценка событий", {
    x: c2x + 0.2, y: cY + 2.6, w: 3.7, h: 0.35,
    fontSize: 12, fontFace: HEADER_FONT, color: ACCENT, bold: true, margin: 0
  });
  s.addText("Оценка событий — отдельный процесс. На этом этапе подключается ИИ-оценка и корреляция.", {
    x: c2x + 0.2, y: cY + 2.95, w: 3.7, h: 1.3,
    fontSize: 11, fontFace: BODY_FONT, color: ICE, margin: 0
  });

  // Col 3: development
  const c3x = 9.2;
  s.addText("РАЗВИТИЕ", {
    x: c3x, y: cY, w: 4.0, h: 0.35,
    fontSize: 10, fontFace: HEADER_FONT, color: ACCENT, bold: true, charSpacing: 6, margin: 0
  });
  const dev = [
    "Замена модели распознавания без правок в логике событий",
    "Подключение ИИ-оценки на стороне событий",
    "Корреляция событий нескольких камер",
    "Запись окна кадров до и после тревоги для разбора",
    "Собственная обученная модель под фойе и багаж",
  ];
  s.addText(
    dev.map((d, i) => ({
      text: d, options: { bullet: true, breakLine: i < dev.length - 1 }
    })),
    {
      x: c3x, y: cY + 0.5, w: 3.7, h: cH - 0.5,
      fontSize: 11, fontFace: BODY_FONT, color: ICE, paraSpaceAfter: 8, margin: 0
    }
  );

  // bottom signature
  s.addShape(pres.shapes.LINE, {
    x: 0.8, y: H - 0.7, w: 4, h: 0, line: { color: ACCENT, width: 1.5 }
  });
  s.addText("Спасибо. Готов к вопросам.", {
    x: 0.8, y: H - 0.6, w: 12, h: 0.4,
    fontSize: 12, fontFace: BODY_FONT, color: MUTED, italic: true, margin: 0
  });
}

pres.writeFile({ fileName: "detection_integra.pptx" }).then(fn => {
  console.log("Saved:", fn);
});
