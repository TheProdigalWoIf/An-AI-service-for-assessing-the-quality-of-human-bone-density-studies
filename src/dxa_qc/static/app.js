const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));

// Названия анатомических областей. Добавлены коды калибровочной модели,
// различающие правое и левое бедро (старый код proximal_femur оставлен —
// его может вернуть резервный правило-ориентированный путь qc.py).
const regionNames = {
  lumbar_spine: 'Поясничный отдел позвоночника',
  proximal_femur_right: 'Проксимальный отдел правого бедра',
  proximal_femur_left: 'Проксимальный отдел левого бедра',
  proximal_femur: 'Проксимальный отдел бедра',
  unknown: 'Не определено',
};
// Названия типов нарушений: пять экспертных критериев текущей модели
// (позвоночник: укладка/ось/артефакты; бедро: ротация/ROI) + таксономия
// калибровочной модели + коды резервных правил qc.py.
const violationNames = {
  none: 'Нарушений не обнаружено',
  // пять экспертных критериев (dxa_qc.ml.VIOLATIONS)
  spine_positioning_incorrect: 'Некорректная укладка позвоночника',
  spine_axis_misalignment: 'Отклонение оси позвоночника',
  spine_artifact_or_object: 'Посторонние предметы, артефакты или наложения',
  hip_positioning_incorrect: 'Ротация/позиционирование бедра некорректны',
  hip_roi_incorrect: 'Область интереса бедра некорректна',
  // коды калибровочной модели
  spine_positioning: 'Некорректная укладка позвоночника',
  spine_axis_over_5deg: 'Ось позвоночника отклонена более чем на 5°',
  spine_foreign_objects: 'Посторонние предметы, артефакты или наложения',
  hip_right_rotation: 'Ротация/позиционирование правого бедра',
  hip_right_roi: 'Область интереса правого бедра некорректна',
  hip_left_rotation: 'Ротация/позиционирование левого бедра',
  hip_left_roi: 'Область интереса левого бедра некорректна',
  suspected_clinical_finding: 'Подозрение на клиническое отклонение',
  // коды резервных правил qc.py
  spine_th12_not_visualized: 'Th12 визуализирован не полностью',
  iliac_crests_not_visualized: 'Подвздошные гребни визуализированы не полностью',
  hip_anatomy_incomplete: 'Неполная визуализация анатомии бедра',
  hip_roi_margin_insufficient: 'Недостаточное поле вокруг ROI',
  hip_overrotation_suspected: 'Подозрение на избыточную ротацию бедра',
  hip_underrotation_suspected: 'Подозрение на недостаточную ротацию бедра',
  foreign_object_suspected: 'Возможный посторонний объект',
  low_contrast: 'Низкий контраст',
  not_dxa_image: 'Не похоже на DXA-снимок',
};

// Сторона снимка (только для бедра; у позвоночника стороны нет).
const sideNames = {none: '—', right: 'Правое', left: 'Левое'};

// Статус итогового решения.
const decisionNames = {final: 'Итоговое решение', needs_review: 'Требует проверки'};

let selectedFiles = [];
const toast = (message) => {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  setTimeout(() => element.classList.remove('show'), 2600);
};

const formatBytes = (bytes) => bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} КБ` : `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
const isDicom = (file) => /\.(dcm|dicom)$/i.test(file.name);
async function init() {
  const health = await fetch('/api/health').then((response) => response.json());
  $('#modelState').classList.toggle('ready', health.model_available);
  $('#modelState').lastChild.textContent = health.model_available ? ' модель готова' : ' резервные правила';
}

function clearSelection() {
  selectedFiles.forEach((item) => item.previewUrl && item.previewUrl.startsWith('blob:') && URL.revokeObjectURL(item.previewUrl));
  selectedFiles = [];
  $('#fileInput').value = '';
  $('#dropzone').hidden = false;
  $('#fileReady').hidden = true;
}

async function loadDicomSelectionPreview(entry) {
  const data = new FormData();
  data.append('file', entry.file);
  try {
    const response = await fetch('/api/upload-preview', {method: 'POST', body: data});
    const payload = await response.json();
    if (!response.ok || selectedFiles[0] !== entry) return;
    entry.previewUrl = payload.preview_data_url;
    $('#uploadPreview').src = entry.previewUrl;
    $('#uploadPreview').hidden = false;
    $('#previewWrap').classList.remove('dicom');
  } catch (_) {
    // Keep the clean DICOM placeholder; full analysis will report read errors.
  }
}

function selectFiles(files) {
  const allowed = /\.(dcm|dicom|jpe?g|png|webp)$/i;
  const incoming = Array.from(files).slice(0, 50);
  const valid = incoming.filter((file) => allowed.test(file.name) && file.size <= 25 * 1024 * 1024);
  if (!valid.length) return toast('Нет подходящих файлов');
  clearSelection();
  selectedFiles = valid.map((file) => ({file, previewUrl: isDicom(file) ? null : URL.createObjectURL(file)}));
  if (valid.length !== incoming.length) toast('Неподдерживаемые или слишком большие файлы пропущены');
  if (files.length > 50) toast('Выбраны первые 50 файлов');
  const first = selectedFiles[0];
  const totalSize = valid.reduce((sum, file) => sum + file.size, 0);
  $('#fileName').textContent = valid.length === 1 ? first.file.name : `Выбрано снимков: ${valid.length}`;
  $('#fileSize').textContent = `${formatBytes(totalSize)} суммарно`;
  $('#fileQueue').innerHTML = valid.slice(0, 6).map((file) => `<span>${esc(file.name)}</span>`).join('') + (valid.length > 6 ? `<span>+ ещё ${valid.length - 6}</span>` : '');
  $('#uploadPreview').src = first.previewUrl || '';
  $('#uploadPreview').hidden = !first.previewUrl;
  $('#previewWrap').classList.toggle('dicom', !first.previewUrl);
  $('#dropzone').hidden = true;
  $('#fileReady').hidden = false;
  $('#analysisResult').hidden = true;
  if (!first.previewUrl) loadDicomSelectionPreview(first);
}

function annotationSvg(annotation) {
  if (!annotation) return '';
  const width = annotation.width, height = annotation.height;
  const [axisStart, axisEnd] = annotation.axis;
  const centerX = (axisStart[0] + axisEnd[0]) / 2;
  const colors = ['#11a7d8', '#a855a5', '#75e882', '#f1df58'];
  const guides = annotation.guides.map((guide, index) => `<line class="guide-line" x1="${guide.start[0]}" y1="${guide.start[1]}" x2="${guide.end[0]}" y2="${guide.end[1]}" stroke="${colors[index % colors.length]}"/><text x="${Math.min(guide.end[0] + 8, width - 90)}" y="${guide.end[1] - 7}" fill="${colors[index % colors.length]}">${guide.label}</text>`).join('');
  const roi = annotation.roi ? `<rect class="roi-box" x="${annotation.roi.x}" y="${annotation.roi.y}" width="${annotation.roi.width}" height="${annotation.roi.height}"/>` : '';
  return `<svg class="qc-overlay" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" aria-label="Автоматическая геометрическая разметка"><line class="reference-line" x1="${centerX}" y1="${height * .08}" x2="${centerX}" y2="${height * .92}"/><line class="axis-line" x1="${axisStart[0]}" y1="${axisStart[1]}" x2="${axisEnd[0]}" y2="${axisEnd[1]}"/>${guides}${roi}<text class="angle-label" x="${Math.max(12, centerX + 12)}" y="${height * .92}">Угол: ${Number(annotation.angle_degrees).toFixed(2).replace('.', ',')}°</text></svg>`;
}

function resultCard(item, index) {
  if (item.error) return `<article class="batch-result error-result"><div><span class="quality-badge bad">ОШИБКА</span><h3>${esc(item.filename)}</h3><p>${esc(item.error)}</p></div></article>`;
  const result = item.result;
  const good = result.quality_class === 0;
  const confidence = Math.round(result.quality_score * 100);
  const violations = result.violation_type.split(';').map((code) => violationNames[code] || code).join(', ');
  const localPreview = selectedFiles[index] && selectedFiles[index].previewUrl;
  const image = localPreview || item.preview_data_url;
  const visual = image ? `<div class="annotated-frame"><img src="${esc(image)}" alt="${esc(item.filename)}">${annotationSvg(item.annotation)}</div>` : '<div class="dicom-visual">DICOM</div>';
  const fields = [
    ['Путь к исследованию', result.path_to_study],
    ['UID исследования', result.study_uid === 'browser_upload' ? 'Загрузка из браузера' : result.study_uid],
    ['UID изображения', result.image_uid],
    ['Анатомическая область', regionNames[result.anatomical_region] || result.anatomical_region],
    ['Сторона', sideNames[result.side] || result.side || '—'],
    ['Класс качества', result.quality_class === 0 ? '0 — качественный снимок' : '1 — нарушение качества'],
    ['Тип нарушения', violations],
    ['Статус решения', decisionNames[result.decision_status] || result.decision_status || '—'],
  ];
  // Отклонение оси позвоночника: угол по остистым отросткам (задание),
  // критерий годности из таблицы разметки — не более 5°.
  if (result.axis_deviation_deg !== null && result.axis_deviation_deg !== undefined) {
    fields.push(['Отклонение оси (норма до 5°)', `${Number(result.axis_deviation_deg).toFixed(2).replace('.', ',')}°`]);
  }
  // Клиническое отклонение (сколиоз, перелом, эндопротез и т.п.) показывается
  // ОТДЕЛЬНО и не является нарушением качества снимка — по правилам задачи.
  if (result.clinical_probability !== null && result.clinical_probability !== undefined
      && Number(result.clinical_probability) >= 0.5) {
    fields.push(['Клиническое отклонение', 'Подозрение на клиническую находку — на качество снимка не влияет']);
  }
  fields.push(
    ['Статус обработки', result.processing_status === 'Success' ? 'Успешно' : 'Ошибка'],
    ['Время обработки', `${result.time_of_processing.toFixed(4).replace('.', ',')} с`],
  );
  // Подпись под снимком: геометрический угол оверлея + отклонение оси по
  // остистым отросткам (если посчитано) + подсказка про увеличение по клику.
  const axisNote = result.axis_deviation_deg !== null && result.axis_deviation_deg !== undefined
    ? ` · отклонение оси ${Number(result.axis_deviation_deg).toFixed(2).replace('.', ',')}°` : '';
  const geometryText = item.annotation ? `Автоматическая геометрическая разметка · расчетный угол ${Number(item.annotation.angle_degrees).toFixed(2).replace('.', ',')}°${axisNote} · нажмите на снимок, чтобы увеличить` : '';
  return `<article class="batch-result"><div class="batch-title"><span>Снимок ${index + 1}</span><h3>${esc(item.filename)}</h3></div><div class="result-grid"><div><div class="result-visual">${visual}</div><div class="geometry-note">${geometryText}</div></div><div class="verdict-card"><span class="quality-badge ${good ? 'good' : 'bad'}">КЛАСС ${result.quality_class}</span><h3>${good ? 'Снимок соответствует критериям' : 'Обнаружено нарушение качества'}</h3><p>${good ? 'Модель не выявила признаков нарушения позиционирования или качества.' : esc(violations)}</p><div class="confidence"><span>Уверенность модели</span><b>${confidence}%</b><div><i class="${good ? 'good' : 'bad'}" style="width:${confidence}%"></i></div></div></div><div class="structured-card"><div class="structured-head"><div><span class="eyebrow">Результат обработки</span><h3>Данные снимка</h3></div><span class="processing-pill success">Успешно</span></div><dl class="result-fields">${fields.map(([name, value]) => `<div><dt>${name}</dt><dd>${esc(value)}</dd></div>`).join('')}</dl></div></div></article>`;
}

function showResults(payload) {
  $('#resultSummary').textContent = `Обработано: ${payload.processed} · ошибок: ${payload.failures}`;
  $('#batchResults').innerHTML = payload.items.map(resultCard).join('');
  $('#analysisResult').hidden = false;
  $('#analysisResult').scrollIntoView({behavior: 'smooth', block: 'start'});
}

async function analyzeUpload(event) {
  event.preventDefault();
  if (!selectedFiles.length) return;
  const button = $('#analyzeButton');
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<i class="spinner"></i> Анализируем: ${selectedFiles.length}…`;
  const data = new FormData();
  selectedFiles.forEach((item) => data.append('files', item.file));
  try {
    const response = await fetch('/api/check-uploads', {method: 'POST', body: data});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Не удалось проверить файлы');
    showResults(payload);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
}

$('#chooseFile').onclick = () => $('#fileInput').click();
$('#dropzone').onclick = (event) => { if (!event.target.closest('button')) $('#fileInput').click(); };
$('#dropzone').onkeydown = (event) => { if (event.key === 'Enter' || event.key === ' ') $('#fileInput').click(); };
$('#fileInput').onchange = (event) => event.target.files.length && selectFiles(event.target.files);
$('#removeFile').onclick = clearSelection;
$('#newCheck').onclick = () => { clearSelection(); $('#analysisResult').hidden = true; window.scrollTo({top: 0, behavior: 'smooth'}); };
$('#uploadForm').onsubmit = analyzeUpload;
['dragenter', 'dragover'].forEach((name) => $('#dropzone').addEventListener(name, (event) => { event.preventDefault(); $('#dropzone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach((name) => $('#dropzone').addEventListener(name, (event) => { event.preventDefault(); $('#dropzone').classList.remove('dragging'); }));
$('#dropzone').addEventListener('drop', (event) => event.dataTransfer.files.length && selectFiles(event.dataTransfer.files));
$('#export').onclick = async () => { const response = await fetch('/api/export.csv', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'}); const blob = await response.blob(); const anchor = document.createElement('a'); anchor.href = URL.createObjectURL(blob); anchor.download = 'dxa_quality_report.csv'; anchor.click(); URL.revokeObjectURL(anchor.href); };
init().catch(() => toast('Сервис недоступен'));

// --- Увеличение снимка по клику (lightbox) ---------------------------------
// Клик по превью (в карточке результата или в зоне загрузки) открывает снимок
// почти на весь экран вместе с геометрическим оверлеем; закрытие — клик по
// фону, крестик или Esc. Внутри клонированного превью id удаляются, чтобы не
// дублировать идентификаторы документа.
function openLightbox(visualEl) {
  const clone = visualEl.cloneNode(true);
  clone.querySelectorAll('[id]').forEach((el) => el.removeAttribute('id'));
  const content = $('#lightboxContent');
  content.innerHTML = '';
  content.appendChild(clone);
  $('#lightbox').hidden = false;
}
function closeLightbox() { $('#lightbox').hidden = true; }
$('#lightbox').addEventListener('click', (event) => { if (!event.target.closest('#lightboxContent')) closeLightbox(); });
$('#lightboxClose').onclick = closeLightbox;
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeLightbox(); });
// Делегирование: карточки результатов добавляются динамически, поэтому
// слушатель живёт на контейнере, а не на каждом превью.
$('#batchResults').addEventListener('click', (event) => {
  const visual = event.target.closest('.result-visual');
  if (visual) openLightbox(visual);
});
$('#previewWrap').addEventListener('click', () => {
  if ($('#uploadPreview').src) openLightbox($('#previewWrap'));
});
